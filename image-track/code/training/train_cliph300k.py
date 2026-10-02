#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_cliph300k.py —— 在 30 万数据上【监督训练】CLIP ViT-H-14（**不用教师**）。

为什么做这个（2026-09-13 决策）：
  · 最终分 = 基底 × 0.75 + 尺寸先验 × 0.25，而基底 = anchor 65% + bigG 20% + s384 15%
    → **anchor 的杠杆 ≈ 0.49δ，是 s384 的 4.4 倍**，而排队中的 s384_str300k 打的是杠杆最低的分量
  · 但 anchor 这条线一直是「自蒸馏复制」：`cliph_rerun` 只用 **21,000 个样本**训练
    （hard=1000 w=4 + soft=20000，`N_CACHE` 那 30000 张因 /hy-tmp/align_cache 不存在而实际是 0），
    与 anchor corr **0.964** → 换它没意义，必须训一个**真正独立、见过 30 万张**的 CLIP-H
  · CLIP-H 实测吞吐 **78.7 张/s** → 30 万张只要 **~1.1 小时/epoch**，成本很低

与 cliph_rerun.py 的三处关键区别：
  1. **不用教师**：直接用 idx_train300k_v2.tsv 的真标签（不碰测试集软标签）
  2. **TFM=stretch 方形拉伸**（历史所有拿正 LB 的模型都用它），而非 crop
  3. 输出带 val_auc（官方 sample 1000 的 20%），用于判断是否欠训

输出：RUN_DIR/train.log（**必须随备份走**）、last.pt / ckpt_*.pt（均 LoRA-only，~60MB）、img_<TAG>.csv
"""
import glob
import io
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

BASE = Path(os.environ.get("BASE_DIR", "/hy-tmp/baycup"))
INDEX = Path(os.environ.get("INDEX", "/hy-tmp/index/idx_train300k_v2.tsv"))
RUN = Path(os.environ.get("RUN_DIR", f"/hy-tmp/baycup/runs/{os.environ.get('TAG', 'cliph300k')}"))
RUN.mkdir(parents=True, exist_ok=True)
CLIP_CACHE = os.environ.get("CLIP_CACHE", str(BASE / "models" / "clip_h14_cache"))
SAMPLE_DIR = Path(os.environ.get("SAMPLE_DIR", str(BASE / "data" / "initial")))
TEST_ROOT = Path(os.environ.get("TEST_ROOT", str(BASE / "data" / "replay")))
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
TAG = os.environ.get("TAG", "cliph300k")
TFM = os.environ.get("TFM", "stretch").lower()      # stretch（默认）| crop
CROP = int(os.environ.get("CROP", "224"))
SHORT = int(os.environ.get("SHORT", "256"))         # 仅 TFM=crop 时生效
BS = int(os.environ.get("BS", "32"))
LR = float(os.environ.get("LR", "7e-5"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
R = int(os.environ.get("R", "32"))
WORKERS = int(os.environ.get("WORKERS", "16"))
CKPT_FRACS = [float(x) for x in os.environ.get("CKPT_FRACS", "0.25,0.5,0.75").split(",") if x]
SMOKE = int(os.environ.get("SMOKE", "0"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))
INIT_LORA = os.environ.get("INIT_LORA", "")         # 可选：从已有 LoRA 热启动
SEED = int(os.environ.get("SEED", "20261013"))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def jpeg_aug(im, rng):
    """JPEG q50-95 质量增强 —— 测试集全部是 JPEG，与历史所有拿分脚本一致。"""
    q = rng.randint(50, 95)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class DS(Dataset):
    """⚠️ 2026-09-12 血泪教训：**绝不能**在读图失败时返回"黑图+标签"。
    那会把错标签静默喂进训练，一次 I/O 抖动就能造成梯度爆炸。正确做法：重试 → 换记录 → 抛错。
    """

    MAX_RETRY = 3

    def __init__(self, records, train, seed):
        self.records, self.train = records, train
        self.rng = random.Random(seed)
        if TFM == "stretch":
            tr = [transforms.Resize((CROP, CROP), interpolation=InterpolationMode.BICUBIC)]
            ev = [transforms.Resize((CROP, CROP), interpolation=InterpolationMode.BICUBIC)]
        else:
            tr = [transforms.Resize(SHORT, interpolation=InterpolationMode.BICUBIC),
                  transforms.RandomCrop(CROP)]
            ev = [transforms.Resize(SHORT, interpolation=InterpolationMode.BICUBIC),
                  transforms.CenterCrop(CROP)]
        self.train_tfm = transforms.Compose(tr + [transforms.ToTensor(),
                                                  transforms.Normalize(CLIP_MEAN, CLIP_STD)])
        self.eval_tfm = transforms.Compose(ev + [transforms.ToTensor(),
                                                 transforms.Normalize(CLIP_MEAN, CLIP_STD)])

    def __len__(self):
        return len(self.records)

    def _load(self, p):
        im = Image.open(p)
        im.load()                        # 真正解码，暴露截断/损坏
        return im.convert("RGB")

    def __getitem__(self, i):
        for attempt in range(self.MAX_RETRY + 1):
            p, label = self.records[i][0], self.records[i][1]
            try:
                im = self._load(p)
                break
            except Exception:
                if attempt == self.MAX_RETRY:
                    i = self.rng.randrange(len(self.records))   # 换一条记录（保持标签正确）
                    p, label = self.records[i][0], self.records[i][1]
                    try:
                        im = self._load(p)
                        break
                    except Exception:
                        raise RuntimeError(f"连续读图失败，数据/磁盘可能有问题: {p}")
        if self.train:
            im = jpeg_aug(im, self.rng)
            if self.rng.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
        x = self.train_tfm(im) if self.train else self.eval_tfm(im)
        return x, np.float32(label)


def load_index():
    recs = []
    with open(INDEX) as f:
        for line in f:
            t = line.rstrip("\n").split("\t")
            if len(t) >= 2 and os.path.exists(t[0]):
                recs.append((t[0], int(t[1])))
    return recs


def load_sample_val():
    csvs = glob.glob(str(SAMPLE_DIR / "**" / "image_sample_data.csv"), recursive=True)
    if not csvs:
        return None, None, None
    smp = pd.read_csv(csvs[0], encoding="utf-8-sig", dtype={"id": str})
    y = smp.label.astype(int).to_numpy()
    root = Path(csvs[0]).parent / "image_sample_data"
    if not root.exists():
        root = Path(csvs[0]).parent / "data" / "image_sample_data"
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, va = next(split.split(np.zeros(len(y)), y))
    return ([(str(root / str(smp.iloc[i].id)), int(y[i])) for i in tr],
            [(str(root / str(smp.iloc[i].id)), int(y[i])) for i in va], y[va])


def build_model():
    import open_clip
    from peft import LoraConfig, get_peft_model
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", cache_dir=CLIP_CACHE)
    vision = model.visual
    embed = getattr(vision, "output_dim", 1280)
    # 与 cliph_rerun 完全一致的 LoRA 配置（open_clip 命名），确保可比
    vision = get_peft_model(vision, LoraConfig(
        r=R, lora_alpha=R, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"])).to(DEV)
    head = torch.nn.Linear(embed, 1).to(DEV)
    return vision, head, embed


def lora_only(vision, head):
    """只存 LoRA + head（~60MB），与 cliph_rerun 的 trainable_state 同格式。

    ⚠️ 2026-09-13 血泪教训：`head` 是 **nn.Linear 模块**，不是张量！
       写成 `head.detach().cpu()` 会在第一次存 checkpoint 时抛
       `AttributeError: 'Linear' object has no attribute 'detach'`
       —— 实测 cliph300k 因此跑到 25% 就崩，新机空转了 6 小时才被发现。
       必须是 `head.state_dict()`（与 train_cliph300k.py 里 head.load_state_dict(st["head"]) 对应）。
    """
    return {"vision": {k: v.detach().cpu() for k, v in vision.state_dict().items() if "lora_" in k},
            "head": {k: v.detach().cpu() for k, v in head.state_dict().items()}}


@torch.inference_mode()
def infer(vision, head, records):
    dl = DataLoader(DS(records, train=False, seed=0), batch_size=BS, shuffle=False, num_workers=WORKERS)
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = vision(x.to(DEV, non_blocking=True)).float()
            out.append(torch.sigmoid(head(f).squeeze(-1).float()).cpu().numpy())
    return np.concatenate(out)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    recs = load_index()
    if MAX_SAMPLES:
        random.Random(SEED).shuffle(recs)
        recs = recs[:MAX_SAMPLES]
    n_real = sum(1 for r in recs if r[1] == 0)
    print(f"device={DEV} TFM={TFM} CROP={CROP} SHORT={SHORT} BS={BS} R={R} EPOCHS={EPOCHS} "
          f"LR={LR} n={len(recs)} real={n_real} fake={len(recs) - n_real}", flush=True)

    smp_tr, val_recs, y_val = load_sample_val()
    if smp_tr:
        recs = recs + smp_tr
        print(f"并入官方样例训练集 {len(smp_tr)} 张；val={len(val_recs)} 张", flush=True)

    vision, head, embed = build_model()
    n_tr = sum(p.numel() for p in vision.parameters() if p.requires_grad)
    print(f"基座 ViT-H-14 embed={embed} 可训参数={n_tr/1e6:.2f}M", flush=True)

    if INIT_LORA and Path(INIT_LORA).exists():
        st = torch.load(INIT_LORA, map_location="cpu", weights_only=False)
        r1 = vision.load_state_dict(st["vision"], strict=False)
        r2 = head.load_state_dict(st["head"])
        print(f"** 热启动自 {INIT_LORA}: unexpected={len(r1.unexpected_keys)} missing(head)={len(r2.missing_keys)}",
              flush=True)

    opt = torch.optim.AdamW(
        [{"params": [p for p in vision.parameters() if p.requires_grad], "lr": LR, "weight_decay": 0.01},
         {"params": head.parameters(), "lr": LR, "weight_decay": 0.01}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS))
    w_real = max(1.0, (len(recs) - n_real) / max(n_real, 1))

    start_ep = 0
    last = RUN / "last.pt"
    if last.exists() and not SMOKE:
        try:
            st = torch.load(last, map_location="cpu", weights_only=False)
            vision.load_state_dict(st["vision"], strict=False)
            head.load_state_dict(st["head"])
            opt.load_state_dict(st["opt"])
            start_ep = int(st.get("epoch", 0))
            print(f"** 断点续训：从 epoch {start_ep} 恢复", flush=True)
        except Exception as e:
            print("续训失败，从头开始:", repr(e)[:150], flush=True)

    best = -1.0
    for ep in range(start_ep, EPOCHS):
        vision.train(); head.train()
        dl = DataLoader(DS(recs, train=True, seed=SEED + ep), batch_size=BS, shuffle=True,
                        num_workers=WORKERS, pin_memory=True, drop_last=True)
        total = len(dl)
        lim = 30 if SMOKE else total
        ckpts = {max(1, min(total - 1, int(f * total))) for f in CKPT_FRACS}
        tot, t0 = 0.0, time.time()
        for step, (x, tgt) in enumerate(dl):
            if step >= lim:
                break
            x = x.to(DEV, non_blocking=True); tgt = tgt.to(DEV, non_blocking=True)
            w = torch.where(tgt > 0.5, torch.ones_like(tgt), torch.full_like(tgt, w_real))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = vision(x).float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    head(f).squeeze(-1).float(), tgt, weight=w)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in vision.parameters() if p.requires_grad] + list(head.parameters()), 1.0)
            opt.step(); tot += float(loss.detach())
            if step % 200 == 0 or step == lim - 1:
                print(f"ep={ep} step={step}/{lim} loss={tot/max(1,step+1):.4f} sec={time.time()-t0:.0f} "
                      f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)
            if step in ckpts:
                st = lora_only(vision, head)
                st.update({"opt": opt.state_dict(), "epoch": ep, "step": step})
                torch.save(st, RUN / f"ckpt_{step}.pt")
                print(f"ckpt saved @ step {step}", flush=True)
                if SMOKE:
                    break
        if SMOKE:
            print("SMOKE DONE (30 steps)", flush=True); return
        sched.step()
        st = lora_only(vision, head); st.update({"opt": opt.state_dict(), "epoch": ep + 1, "step": 0})
        torch.save(st, last)
        if val_recs:
            vision.eval()
            auc = roc_auc_score(y_val, infer(vision, head, val_recs))
            print(f"ep={ep} train_loss={tot/max(1,lim):.4f} val_auc={auc:.6f}", flush=True)
        else:
            auc = -1.0
        if auc > best:
            best = auc
            torch.save(lora_only(vision, head), RUN / "best.pt")
            print("best.pt saved", flush=True)

    # ---- 最终：官方 20000 测试图打分 ----
    bp = RUN / "best.pt"
    if bp.exists():
        st = torch.load(bp, map_location="cpu", weights_only=False)
        vision.load_state_dict(st["vision"], strict=False); head.load_state_dict(st["head"])
    vision.eval()
    ids = pd.read_csv(TEST_ROOT / "image_test.csv", encoding="utf-8-sig", dtype={"id": str}).id.tolist()
    scores = infer(vision, head, [(str(TEST_ROOT / "image_test" / i), 0) for i in ids])
    dst = RUN / f"img_{TAG}.csv"
    pd.DataFrame({"id": ids, "score": scores}).to_csv(dst, index=False)
    print(f"DONE {TAG} best_val_auc={best:.6f} output={dst} n={len(scores)}", flush=True)


if __name__ == "__main__":
    main()
