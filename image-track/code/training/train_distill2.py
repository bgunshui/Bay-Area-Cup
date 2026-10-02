#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_distill2.py —— 测试集**转导蒸馏**：把「当前最强混合」教进一个 CLIP-H-14 + LoRA 学生。

来源与动机
----------
`cliph_rerun.py` 用「anchor 当 teacher + sample1000 真标签 + 测试集 20000 软标签」训出的学生，
是当前最强基底里 **权重最大（65%）** 的分量（`0.65·rank(cliph_rerun) + 0.20·rank(bigg_rerun) + 0.15·rank(s384r)`）。
本脚本 = **同配方**，但 teacher 换成**现在更强的混合**（LB 0.914498 vs anchor 的 0.900099）。

与 cliph_rerun.py 的差异（每条都有理由）
---------------------------------------
1. 路径显式化（`TEST_CSV/TEST_ROOT/SAMPLE_CSV/SAMPLE_DIR`）—— 新机有两套目录布局（`/root/sj-tmp/baycup/...`
   与 `/root/sj-tmp/baycup/ds300k/baycup/...`），靠 BASE 拼接必错。
2. `TFM` 可选 `stretch` / `crop`。历史上**拿过正 LB 的分量全部是 stretch**，而 cliph 系一直是 crop
   —— 这是个至今未决的单变量问题，顺手测掉。
3. **干净 val**：按 `probe_cliph.py` 完全相同的划分（`StratifiedShuffleSplit(test_size=0.2, random_state=20260925)`）
   留出 200 张 sample **不参与训练**，每个 epoch 末打印 `val_auc` —— 与所有其他探针的 val 逐图可比。
4. `INIT_FROM`：可从 `cliph_rerun/last.pt` 热启动，而不是每次从零学。
5. 冒烟会**真正走到存盘分支**（`CKPT_EVERY` 传小值）—— §0.9 第 7 条，冒烟走不到的代码等于没冒烟。

⚠️ 分辨率守卫（§0.9 第 9 条）：open_clip 的 `VisionTransformer._embeds()` **不做位置编码插值**，
   直接 `x + positional_embedding`。ViT-H-14 的 patch=14、位置编码 16×16(+CLS) → **CROP 必须 == 224**。
   想换分辨率必须先插值位置编码，且训练/推理/探针三处都得改 —— 本脚本直接抛错拒绝，不做半套。

用法
----
  TEACHER=/path/teacher.csv TAG=cliph_d1 RUN_DIR=... python3 -u train_distill2.py
输出
----
  RUN_DIR/last.pt（LoRA-only + head + optimizer + epoch）、RUN_DIR/img_{TAG}.csv（20000 行）
"""
import glob
import io
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_B = "/root/sj-tmp/baycup/ds300k/baycup"
TEST_CSV = Path(os.environ.get("TEST_CSV", f"{DEFAULT_B}/data/official_test/image_test.csv"))
TEST_ROOT = Path(os.environ.get("TEST_ROOT", f"{DEFAULT_B}/data/official_test"))
SAMPLE_DIR = Path(os.environ.get("SAMPLE_DIR", f"{DEFAULT_B}/data/sample"))
SAMPLE_CSV = os.environ.get("SAMPLE_CSV", "")
TEACHER = Path(os.environ["TEACHER"])
RUN = Path(os.environ.get("RUN_DIR", "/root/sj-tmp/baycup/ds300k/runs/cliph_distill"))
CLIP_CACHE = os.environ.get("CLIP_CACHE", "/root/sj-tmp/baycup/models/clip_h14_cache")
INIT_FROM = os.environ.get("INIT_FROM", "")
TAG = os.environ.get("TAG", "cliph_distill")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TFM = os.environ.get("TFM", "crop").lower()
SHORT = int(os.environ.get("SHORT", "256"))
CROP = int(os.environ.get("CROP", "224"))
BS = int(os.environ.get("BS", "32"))
LR = float(os.environ.get("LR", "1e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
R = int(os.environ.get("R", "32"))
SEED = int(os.environ.get("SEED", "20261003"))
HARD_W = float(os.environ.get("HARD_W", "4.0"))
# 可选：再并入一份硬标签索引（Tiny 28.8k 等）。见 main() 里的说明。
EXTRA_INDEX = os.environ.get("EXTRA_INDEX", "")
EXTRA_W = float(os.environ.get("EXTRA_W", "1.0"))
EXTRA_MAX = int(os.environ.get("EXTRA_MAX", "0"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "500"))
RESUME = int(os.environ.get("RESUME", "1"))
SMOKE = int(os.environ.get("SMOKE", "0"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
# ⚠️ 2026-09-13 补：原来本脚本**没有** MAX_SAMPLES（只有 MAX_STEPS），
#    而 run_distill_queue.sh 一直在传 `MAX_SAMPLES=1200` —— 被静默忽略（冒烟靠 lim=30 兜住了）。
#    我在加 EXTRA_INDEX 时引用了它 → NameError，整个脚本起不来。
#    教训：**新增一个环境变量前，先确认它真的被定义过**（grep 一遍），别假设同名变量在别的脚本里有这里就有。
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))
WORKERS = int(os.environ.get("WORKERS", "10"))
VAL_HOLD = int(os.environ.get("VAL_HOLD", "200"))
SOFT_GAIN = float(os.environ.get("SOFT_GAIN", "4.0"))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def jpeg_aug(im, rng):
    q = rng.randint(50, 95)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_tfm(train):
    if TFM == "stretch":
        t = [transforms.Resize((CROP, CROP), interpolation=InterpolationMode.BICUBIC)]
    elif TFM == "crop":
        t = [transforms.Resize(SHORT, interpolation=InterpolationMode.BICUBIC),
             transforms.RandomCrop(CROP) if train else transforms.CenterCrop(CROP)]
    else:
        raise ValueError(f"未知 TFM={TFM}（只支持 crop / stretch）")
    return transforms.Compose(t + [transforms.ToTensor(),
                                   transforms.Normalize(CLIP_MEAN, CLIP_STD)])


class DS(Dataset):
    def __init__(self, items, train, seed=0):
        self.items, self.train = items, train
        self.tfm = make_tfm(train)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, tgt, w = self.items[i]
        im, err = None, None
        for _ in range(3):                      # 绝不返回黑图（会静默污染训练/val）
            try:
                im = Image.open(p).convert("RGB")
                im.load()
                break
            except Exception as e:
                err = e
        if im is None:
            raise RuntimeError(f"读图连续失败: {p} ({err})")
        if self.train:
            im = jpeg_aug(im, self.rng)
            if self.rng.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return self.tfm(im), np.float32(tgt), np.float32(w)


def lora_state(vision):
    """只存 LoRA 张量（⚠️ state_dict() 过滤，不是对 module 直接 .detach() —— §0.9 第 8 条）。"""
    return {k: v.detach().cpu() for k, v in vision.state_dict().items() if "lora_" in k}


@torch.inference_mode()
def predict(vision, head, items):
    vision.eval(); head.eval()
    dl = DataLoader(DS(items, train=False), batch_size=BS, shuffle=False, num_workers=WORKERS)
    out = []
    for x, _, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = vision(x.to(DEV, non_blocking=True)).float()
            out.append(torch.sigmoid(head(f).squeeze(-1).float()).cpu().numpy())
    return np.concatenate(out)


def build():
    import open_clip
    from peft import LoraConfig, get_peft_model
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", cache_dir=CLIP_CACHE)
    vision = model.visual
    embed = getattr(vision, "output_dim", 1280)

    # ---- 分辨率守卫：位置编码不插值，CROP 必须等于原生输入边长 ----
    vit = None
    for _, m in vision.named_modules():
        if hasattr(m, "positional_embedding") and hasattr(m, "conv1"):
            vit = m
            break
    if vit is None:
        raise RuntimeError("找不到 open_clip ViT 主体，无法校验分辨率")
    n_pe = int(vit.positional_embedding.shape[0])
    ks = getattr(vit.conv1, "kernel_size", 14)
    patch = int(ks[0]) if isinstance(ks, (tuple, list)) else int(ks)
    grid = int(round((n_pe - 1) ** 0.5))
    native = grid * patch
    print(f"基座 ViT-H-14 embed={embed} patch={patch} 位置编码 {grid}x{grid} → 原生输入 {native}", flush=True)
    if CROP != native:
        raise RuntimeError(f"CROP={CROP} != 原生 {native}：open_clip 不插值位置编码，会直接崩或错。"
                           f"换分辨率必须先插值位置编码，训练/推理/探针三处同改。")

    lcfg = LoraConfig(r=R, lora_alpha=R, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"])
    vision = get_peft_model(vision, lcfg).to(DEV)
    head = torch.nn.Linear(embed, 1).to(DEV)
    print(f"可训参数={sum(p.numel() for p in vision.parameters() if p.requires_grad)/1e6:.2f}M", flush=True)
    return vision, head


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    RUN.mkdir(parents=True, exist_ok=True)
    print(f"[{TAG}] TFM={TFM} SHORT={SHORT} CROP={CROP} BS={BS} LR={LR} EPOCHS={EPOCHS} R={R} "
          f"SMOKE={SMOKE} VAL_HOLD={VAL_HOLD}", flush=True)
    vision, head = build()

    if INIT_FROM:
        st0 = torch.load(INIT_FROM, map_location="cpu", weights_only=False)
        ls = st0["vision"] if "vision" in st0 else st0["lora"]
        hit = vision.load_state_dict(ls, strict=False)
        n_lora = sum(1 for k in ls if "lora_" in k)
        print(f"INIT_FROM={Path(INIT_FROM).name} 映射={len(ls) - len(hit.unexpected_keys)}/{n_lora} "
              f"unexpected={len(hit.unexpected_keys)}", flush=True)
        if len(hit.unexpected_keys) or (n_lora and len(ls) - len(hit.unexpected_keys) < n_lora):
            raise RuntimeError(f"INIT_FROM 键名不匹配（§0.9 第 10 条），拒绝静默丢权重："
                               f"{hit.unexpected_keys[:3]}")
        if "head" in st0:
            head.load_state_dict(st0["head"])
            print("  head 也已载入", flush=True)

    # ---- 数据 ----
    rep = pd.read_csv(TEST_CSV, encoding="utf-8-sig", dtype={"id": str})
    RIDS = rep.id.astype(str).tolist()
    t = pd.read_csv(TEACHER, dtype={"id": str})
    t["id"] = t.id.astype(str)
    if t.id.tolist() != RIDS:
        t = t.set_index("id").loc[RIDS].reset_index()
    t_rank = rankdata(t.score.to_numpy(), method="average") / (len(t) + 1.0)
    soft = 1.0 / (1.0 + np.exp(-SOFT_GAIN * (t_rank - 0.5)))
    print(f"teacher={TEACHER.name} soft mean={soft.mean():.3f} std={soft.std():.3f}", flush=True)

    scsv = SAMPLE_CSV or glob.glob(str(SAMPLE_DIR / "**" / "image_sample_data.csv"), recursive=True)[0]
    smp = pd.read_csv(scsv, encoding="utf-8-sig", dtype={"id": str})
    sroot = Path(scsv).parent / "image_sample_data"
    y = smp.label.astype(int).to_numpy()
    # 与 probe_cliph.py / probe_ckpt.py 完全相同的划分种子 → val 逐图可比
    sp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=20260925)
    _, va = next(sp.split(np.zeros(len(y)), y))
    vset = set(int(i) for i in va)
    tri = [i for i in range(len(y)) if i not in vset] if VAL_HOLD else list(range(len(y)))
    print(f"sample 共 {len(y)}：训练 {len(tri)}（w={HARD_W}），val {len(vset)}（不参与训练）", flush=True)

    items = [(str(sroot / str(smp.iloc[i].id)), float(y[i]), HARD_W) for i in tri]
    items += [(str(TEST_ROOT / "image_test" / i), float(s), 1.0) for i, s in zip(RIDS, soft)]
    # 可选：再并入一份**硬标签**索引（如 idx_tiny.tsv = Tiny-GenImage 28.8k）。
    # 为什么：`cliph_rerun` 的原配方里**没有 Tiny**，只有测试集软标签 + sample 硬标签；
    #   而 2026-09-13 实测 `cliph300k`（302k 真标签）探针 val = 0.99305、与 cliph_rerun 相关仅 0.81
    #   → "真标签监督"确实带来独立信号。Tiny 是唯一被 LB 验证过的数据（§⑪），值得并进来。
    if EXTRA_INDEX:
        lines = [l for l in Path(EXTRA_INDEX).read_text().splitlines() if l.strip()]
        if EXTRA_MAX:
            random.Random(SEED).shuffle(lines); lines = lines[:EXTRA_MAX]
        ex = [(l.split("\t")[0], float(l.split("\t")[1]), EXTRA_W) for l in lines]
        alive = [r for r in ex if Path(r[0]).exists()]
        if len(alive) != len(ex):
            print(f"  [warn] EXTRA_INDEX 丢弃 {len(ex)-len(alive)}/{len(ex)} 条不存在路径", flush=True)
        items += alive
        n_r = sum(1 for r in alive if r[1] < 0.5)
        print(f"  并入 EXTRA_INDEX {len(alive)} 张（w={EXTRA_W}，real={n_r} fake={len(alive)-n_r}）"
              f" ← {Path(EXTRA_INDEX).name}", flush=True)
    if MAX_SAMPLES:
        random.Random(SEED).shuffle(items)          # 冒烟时打乱再截断，避免只取到某一类
        items = items[:MAX_SAMPLES]
    print(f"训练样本 {len(items)}（hard={len(tri)}，soft={len(RIDS)}）", flush=True)

    vitems = [(str(sroot / str(smp.iloc[i].id)), int(y[i]), 1.0) for i in sorted(vset)]

    params = [p for p in vision.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(
        [{"params": [p for p in vision.parameters() if p.requires_grad], "lr": LR, "weight_decay": 0.01},
         {"params": list(head.parameters()), "lr": LR, "weight_decay": 0.01}])

    # ---- 断点续训 ----
    last = RUN / "last.pt"
    start_ep = 0
    if RESUME and last.exists() and not INIT_FROM:
        st = torch.load(last, map_location="cpu", weights_only=False)
        vision.load_state_dict(st["vision"], strict=False)
        head.load_state_dict(st["head"])
        try:
            opt.load_state_dict(st["opt"])
        except Exception as e:
            print("optimizer 恢复失败(忽略):", repr(e)[:100], flush=True)
        start_ep = int(st.get("epoch", 0))
        print(f"** 断点续训: 从 epoch {start_ep} 恢复", flush=True)

    for ep in range(start_ep, EPOCHS):
        vision.train(); head.train()
        dl = DataLoader(DS(items, train=True, seed=SEED + ep), batch_size=BS, shuffle=True,
                        num_workers=WORKERS, pin_memory=True, drop_last=True)
        t0, tot, n = time.time(), 0.0, 0
        lim = min(len(dl), 30) if SMOKE else (MAX_STEPS or len(dl))
        for step, (x, tg, w) in enumerate(dl):
            if step >= lim:
                break
            x = x.to(DEV, non_blocking=True); tg = tg.to(DEV); w = w.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = vision(x).float()
                logit = head(f).squeeze(-1).float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, tg, weight=w)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss.detach()); n += 1
            if step % 200 == 0 or step == lim - 1:
                print(f"ep={ep} step={step}/{lim} loss={tot/max(n,1):.4f} sec={time.time()-t0:.0f} "
                      f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)
            if CKPT_EVERY and step > 0 and step % CKPT_EVERY == 0:
                torch.save({"vision": lora_state(vision), "head": head.state_dict(),
                            "opt": opt.state_dict(), "epoch": ep}, last)
                print(f"ckpt saved @ ep{ep} step{step} -> {last}", flush=True)

        # ---- epoch 末：val ----
        vs = predict(vision, head, vitems)
        auc = roc_auc_score([v[1] for v in vitems], vs)
        print(f"** ep={ep} val_auc={auc:.6f} (n={len(vitems)})", flush=True)
        if SMOKE:
            # 顺手覆盖"写 CSV"这条交付路径（§0.9 第 7 条：冒烟要走遍所有分支）
            ss = predict(vision, head, [(str(TEST_ROOT / "image_test" / i), 0.0, 1.0) for i in RIDS[:8]])
            pd.DataFrame({"id": RIDS[:8], "score": ss}).to_csv(RUN / "smoke_write.csv", index=False)
            print(f"SMOKE DONE（含 CSV 写入自检，8 行）", flush=True)
            return
        torch.save({"vision": lora_state(vision), "head": head.state_dict(),
                    "opt": opt.state_dict(), "epoch": ep + 1}, last)
        print(f"** epoch {ep} 完成并保存 ({last})", flush=True)

    # ---- 测试集推理 ----
    sc = predict(vision, head, [(str(TEST_ROOT / "image_test" / i), 0.0, 1.0) for i in RIDS])
    dst = RUN / f"img_{TAG}.csv"
    pd.DataFrame({"id": RIDS, "score": sc}).to_csv(dst, index=False)
    print(f"DONE {TAG} output={dst} n={len(sc)} mean={sc.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
