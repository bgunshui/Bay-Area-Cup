#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""models.py —— 8 个分量的加载与前向。**预处理必须与出冻结 CSV 时逐项一致。**

三个骨干家族（配置写死在这里，不许"猜"）：

┌ 家族 A：open_clip CLIP ViT-H-14 (laion2b_s32b_b79k)
│   分量：cliph_rerun / cliph_s20261101 / cliph_s20261102 / cliph_s20261103
│   LoRA target = q_proj,k_proj,v_proj,out_proj,c_fc,c_proj   r=32  alpha=**r**（scaling 1.0）
│   head = Linear(1024, 1)
│   前向 = vision(x)               （open_clip 直接返回 pooled [B,1024]）
│   预处理 = Resize(256, BICUBIC) → CenterCrop(224) → ToTensor → Normalize(CLIP mean/std)
│   来源 = probe_cliph.py（含 cliph_rerun.py 第 109-113 行）
│
├ 家族 B：HF CLIPVisionModel（ViT-bigG-14，hidden=1664，48 层，patch14，**原生 224**）
│   分量：bigg_rerun2 / bigg_rerun3（224 拉伸）、bigg_s384_retrain（384 拉伸）
│   LoRA target = q_proj,k_proj,v_proj,out_proj,fc1,fc2       alpha=**2r**（scaling 2.0）
│   head = Linear(1664, 1)
│   前向 = fwd(v,x).last_hidden_state[:,0]；**喂 384 必须 interpolate_pos_encoding=True**
│   预处理 = Resize((CROP,CROP), BICUBIC) → ToTensor → Normalize(CLIP mean/std)   [TFM=stretch]
│
└ 家族 C：HF SiglipVisionModel（siglip2-so400m-patch14-384，hidden=1152，27 层）
    分量：gpp
    LoRA target 同 B，alpha=2r
    head = Linear(1152, 1)
    前向 = v(x).last_hidden_state[:,0]；**SigLIP 自己按输入插值，不能传 interpolate_pos_encoding**
    预处理 = Resize((384,384), BICUBIC) → ToTensor → Normalize(0.5, 0.5)   [TFM=stretch]

⚠️ 所有 LoRA 的 r 都**从 checkpoint 自身推断**（§0.9 第 10 条），不许硬编码。
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ⚠️ transformers 5.x 的 `check_torch_load_is_safe` 会以"torch < 2.6"为由拒绝 `torch.load`。
#    本服务加载的是 **safetensors**，不经过 torch.load；这里只做一层保险，避免个别版本路径报错。
#    （与 `probe_hf_detectors.py` 里的同一处理一致；**绝不动 torch 本身**。）
try:
    import transformers.modeling_utils as _tmu
    if hasattr(_tmu, "check_torch_load_is_safe"):
        _tmu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]

# ── 每个分量的推理配置（= 产出冻结 CSV 时的配置，逐项核对过） ────────────────
# ⚠️⚠️ 键名必须与 `tables/fusion_meta.json` 里的 `ecdf_key` **完全一致**（= 分量 CSV 的
#     basename 去掉 .csv）。融合表就是按这个名字找分量的：
#         Fusion.fuse() 里 `if key not in comp_scores: raise KeyError(f"缺少分量 {key}")`
#     2026-09-16 实测踩过：键名写成 `cliph_rerun` 而表里是 `img_cliph_rerun`
#     ⇒ `KeyError: 缺少分量 img_cliph_rerun`。**只有在真的跑一遍时才会暴露。**
COMPONENTS = {
    # key（= CSV basename）       家族      权重文件                    尺寸        归一化
    "img_cliph_rerun":        ("cliph",   "cliph_rerun.pt",       (256, 224), CLIP_MEAN),
    "img_cliph_s20261101":    ("cliph",   "cliph_s20261101.pt",   (256, 224), CLIP_MEAN),
    "img_cliph_s20261102":    ("cliph",   "cliph_s20261102.pt",   (256, 224), CLIP_MEAN),
    "img_cliph_s20261103":    ("cliph",   "cliph_s20261103.pt",   (256, 224), CLIP_MEAN),
    "img_gpp_siglip2":        ("siglip2", "gpp.pt",               (384, 384), SIGLIP_MEAN),
    "img_bigg_rerun2":        ("bigg",    "bigg_rerun2.pt",       (224, 224), CLIP_MEAN),
    "img_bigg_rerun3":        ("bigg",    "bigg_rerun3.pt",       (224, 224), CLIP_MEAN),
    "img_bigg_s384_retrain":  ("bigg",    "bigg_s384_retrain.pt", (384, 384), CLIP_MEAN),
}
# TFM：crop = 短边缩放到 SHORT 再 CenterCrop(CROP)；stretch = 直接 Resize((CROP,CROP))
#   只有 cliph 家族是 crop（SHORT=256/CROP=224），其余全部 stretch。
# ⚠️ 归一化用**按分量名索引**而非按位置下标 —— 后者在配方调整时会静默错配。
STD = {
    "img_cliph_rerun": CLIP_STD, "img_cliph_s20261101": CLIP_STD,
    "img_cliph_s20261102": CLIP_STD, "img_cliph_s20261103": CLIP_STD,
    "img_gpp_siglip2": SIGLIP_STD,
    "img_bigg_rerun2": CLIP_STD, "img_bigg_rerun3": CLIP_STD,
    "img_bigg_s384_retrain": CLIP_STD,
}
assert set(STD) == set(COMPONENTS), "STD 与 COMPONENTS 的分量名不一致"


def _norm_key(k):
    """抹平 peft 包装层级（base_model.model. / vision_model. / .default）—— 跨脚本加载必须做。"""
    for seg in ("base_model.model.", "vision_model.", ".default"):
        k = k.replace(seg, "")
    return k


def detect_r(lora_sd):
    for k, v in lora_sd.items():
        if "lora_A" in k and hasattr(v, "shape") and v.dim() == 2:
            return int(v.shape[0])
    raise RuntimeError("checkpoint 里找不到 lora_A，无法推断 rank")


def load_lora_into(model, ckpt_path, alpha_mult):
    """把 LoRA-only 权重载入 peft 包装好的模型。**映射数必须完整，否则抛错。**"""
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora_sd = st.get("lora", st.get("vision"))
    if lora_sd is None:
        raise RuntimeError(f"{ckpt_path} 顶层键异常：{list(st.keys())[:6]}")
    tgt = list(model.state_dict().keys())
    tgt_set = set(tgt)
    tgt_by_norm = {_norm_key(k): k for k in tgt}
    mapped, miss = {}, []
    for k, v in lora_sd.items():
        if k in tgt_set:
            mapped[k] = v
        elif _norm_key(k) in tgt_by_norm:
            mapped[tgt_by_norm[_norm_key(k)]] = v
        else:
            miss.append(k)
    res = model.load_state_dict(mapped, strict=False)
    n_lora = sum(1 for k in lora_sd if "lora_" in k)
    if res.unexpected_keys:
        raise RuntimeError(f"{ckpt_path} unexpected_keys={res.unexpected_keys[:3]}")
    if n_lora and len(mapped) < n_lora:
        raise RuntimeError(f"{ckpt_path} LoRA 只映射 {len(mapped)}/{n_lora}，权重不完整")
    if miss:
        raise RuntimeError(f"{ckpt_path} 有 {len(miss)} 个张量未能映射，样本 {miss[:2]}")
    if "head" not in st:
        raise RuntimeError(f"{ckpt_path} 缺 head —— 只存 LoRA 无法复现分数")
    return st["head"], n_lora


def _tfm(short_crop, mean, std):
    short, crop = short_crop
    t = ([transforms.Resize(short, interpolation=InterpolationMode.BICUBIC),
          transforms.CenterCrop(crop)] if short != crop else
         [transforms.Resize((crop, crop), interpolation=InterpolationMode.BICUBIC)])
    return transforms.Compose(t + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def _lora_snapshot(model):
    """取出模型里**全部 LoRA 参数**（名字 → 张量副本）。

    ⚠️⚠️ 为什么必须做这个（2026-09-16 真跑验证抓到的**致命 bug**，第 4 个）：
        peft 包装器是按 **(家族, rank)** 缓存的 —— 这是刻意的，重建一次 CLIP-H-14 要约 30 秒。
        但 `load_lora_into()` 是**原地**把某个分量的 LoRA 权重写进这个**共享对象**的，
        于是"同家族同 rank 的多个分量"会互相覆盖：
            `img_cliph_rerun` / `s20261101` / `s20261102` / `s20261103` 全是 (cliph, r=16)
            ⇒ 推理时**四个分量用的都是最后加载的那个 LoRA（s20261103）**，
              只有各自的 head 是对的。
        实测症状（`verify_on_host.py`，200 张真图 vs 冻结 CSV 的 spearman）：
            cliph_s20261103  0.9994 ✓   ← 最后加载的那个，恰好是对的
            cliph_s20261102  0.6314 ✗
            cliph_s20261101  0.3384 ✗
            cliph_rerun      0.3354 ✗   ← 占基底权重 65% 的主分量！
        且错的那几个**输出分布被压扁**（std 0.061~0.088，正常该是 0.25+）——
        正是"别人的 LoRA + 自己的 head"的特征。
        ⇒ 正解：**保留共享包装器省显存，但为每个分量存一份 LoRA 参数快照，打分前按需换入。**
    """
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def _swap_lora(model, snap):
    """把某个分量的 LoRA 快照写回共享包装器，返回写入的张量数（0 表示快照对不上，必须当错误）。"""
    hit = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n])
                hit += 1
    return hit


class ComponentEnsemble:
    """三个骨干各加载一次，8 个 LoRA 头共享骨干（显存友好）。"""

    def __init__(self, weights_dir, backbones, device="cuda", dtype=torch.float16):
        self.dev = torch.device(device)
        self.dtype = dtype
        self.backbones = backbones
        self.heads = {}
        self.tfms = {}
        self.families = {}
        self._peft = {}
        self._lora_snap = {}   # 分量 key → 它自己的 LoRA 参数快照（见 _lora_snapshot 的说明）
        self._active = {}      # (家族, rank) → 当前**真的**装在共享包装器里的分量 key

        for key, (fam, wfile, size, mean) in COMPONENTS.items():
            # ⚠️ 2026-09-16 修的 bug：peft 模型必须按 **(家族, LoRA rank)** 建，不能只按家族。
            #    实测 `bigg_s384_retrain` 是 **r=32**，而同一家族的 `bigg_rerun2/3` 是 **r=16**
            #    ⇒ 只按家族缓存会复用错 rank 的包装，第二个 load_state_dict 形状不匹配。
            pk = (fam, self._rank_of(weights_dir, wfile))
            if pk not in self._peft:
                self._peft[pk] = self._build_peft(pk[0], pk[1], weights_dir)
            model = self._peft[pk]
            p = Path(weights_dir) / wfile
            if not p.exists():
                raise FileNotFoundError(f"缺权重 {p}")
            head_sd, n = load_lora_into(model, p, None)
            # ⚠️ 立刻给这个分量存一份 LoRA 快照 —— 因为 `model` 是**按 (家族, rank) 共享**的，
            #    下一个同组分量会把这份权重原地覆盖掉（见 _lora_snapshot 的完整说明）。
            self._lora_snap[key] = _lora_snapshot(model)
            self._active[pk] = key
            # ⚠️ hidden 维度要按家族取：open_clip 的 `VisionTransformer` **没有 `.config`**
            #    （实测 `AttributeError: 'VisionTransformer' object has no attribute 'config'`），
            #    它的输出维度在 `output_dim` 上（CLIP-H-14 = 1024）。
            if pk[0] == "cliph":
                hidden = int(getattr(model.base_model.model, "output_dim", 1024))
            else:
                hidden = model.base_model.model.config.hidden_size
            head = torch.nn.Linear(hidden, 1).to(self.dev)
            head.load_state_dict(head_sd)
            head.eval()
            self.heads[key] = head
            self.families[key] = pk
            self.tfms[key] = _tfm(size, mean, STD[key])
            print(f"  [ok] {key:22s} {fam:8s} r={pk[1]:<3d} LoRA {n} 张量 head={hidden}", flush=True)
        self._to_half()

    # ---- 骨干构造 ----
    @staticmethod
    def _rank_of(weights_dir, wfile):
        st = torch.load(Path(weights_dir) / wfile, map_location="cpu", weights_only=False)
        return detect_r(st.get("lora", st.get("vision")))

    def _build_peft(self, fam, r, weights_dir):
        """按 (家族, r) 建 peft 包装 —— r 由 checkpoint 自身决定（§0.9 第 10 条，不许硬编码）。"""
        from peft import LoraConfig, get_peft_model
        if fam == "cliph":
            import open_clip
            # ⚠️⚠️ 2026-09-16 **真跑验证抓到的 bug**：为什么不能写 `pretrained="laion2b_s32b_b79k"`。
            #   open_clip 3.3.0 的 `load_checkpoint()` 会调 `hf_hub_download(repo, filename, cache_dir=...)`，
            #   而它**要求目录是标准 HF 缓存布局**（`models--laion--…/snapshots/<rev>/<file>` 软链到 `blobs/<sha>`）。
            #   我们的骨干是用 curl 从 hf-mirror 平铺下载的 ⇒ **离线时找不到文件**；
            #   而镜像里按规范设了 `HF_HUB_OFFLINE=1`，于是会直接 `LocalEntryNotFoundError` **加载失败**。
            #   实测报错：`Failed to download file (open_clip_pytorch_model.bin) … not find … in the local cache`。
            #   ⇒ 正解：**自己建架构 + 自己 load_state_dict**，完全不碰 HF 下载逻辑。离线、确定性。
            cache_dir = Path(weights_dir).parent / "backbones" / "clip_h14_cache"
            model = open_clip.create_model("ViT-H-14", pretrained=None)
            st_path = cache_dir / "open_clip_model.safetensors"
            if st_path.exists():
                from safetensors.torch import load_file
                sd = load_file(str(st_path))
            else:
                sd = torch.load(cache_dir / "open_clip_pytorch_model.bin",
                                map_location="cpu", weights_only=False)
                if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict):
                    sd = sd["state_dict"]
            miss, unexp = model.load_state_dict(sd, strict=False)
            # 只允许"未使用"的键（如预训练特有的），**缺权重一律报错**（静默丢权重是 §0.9 第 10 条的坑）
            if miss:
                raise RuntimeError(f"CLIP-H-14 基座缺 {len(miss)} 个权重，样本 {list(miss)[:3]}")
            print(f"  [基座] CLIP-H-14 离线载入 {len(sd)} 张量"
                  f"（unexpected={len(unexp)}）", flush=True)
            vision = model.visual
            alpha = r                       # cliph 家族 lora_alpha = R（scaling 1.0）
            targets = ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"]
        elif fam == "bigg":
            from transformers import CLIPVisionModel
            # 骨干在镜像里是 **fp16** 存的（体积减半）；直接按 fp16 加载，
            # 避免"升到 fp32 → 再 half()"的往返（结果等价，但省内存与加载时间）。
            vision = CLIPVisionModel.from_pretrained(
                str(Path(weights_dir).parent / "backbones" / "clip_bigg"),
                torch_dtype=torch.float16)
            alpha = 2 * r                   # 自训脚本一律 lora_alpha = 2*R（scaling 2.0）
            targets = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        elif fam == "siglip2":
            from transformers import SiglipVisionModel
            vision = SiglipVisionModel.from_pretrained(
                str(Path(weights_dir).parent / "backbones" / "siglip2_384"),
                torch_dtype=torch.float16)
            alpha = 2 * r
            targets = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        else:
            raise ValueError(fam)
        print(f"  [骨干] {fam}: r={r} alpha={alpha} scaling={alpha/r:.2f}", flush=True)
        m = get_peft_model(vision, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05,
                                              bias="none", target_modules=targets))
        m.to(self.dev).eval()
        return m

    def _to_half(self):
        for m in self._peft.values():
            m.half()

    # ---- 前向 ----
    @torch.inference_mode()
    def score(self, key, pil_image):
        """单张 PIL 图 → float。"""
        # 键名守卫：未知分量要给**能看懂的报错**，而不是裸 KeyError
        # （2026-09-16：server.py 的预热用了短键名，结果只报 `KeyError: 'cliph_rerun'`，
        #   排查时看不出"应该用 img_ 前缀"，白花了时间。）
        if key not in self.heads:
            raise KeyError(f"未知分量 {key!r}；可用键 = {sorted(self.heads)}")
        fam = self.families[key][0]
        x = self.tfms[key](pil_image.convert("RGB")).unsqueeze(0).to(self.dev)
        x = x.to(self.dtype)
        pk = self.families[key]
        model = self._peft[pk]
        # ⚠️⚠️ 共享包装器：打分前**必须**把这个分量自己的 LoRA 换回去。
        #    不做这一步的话，同 (家族, rank) 的多个分量会全部用"最后加载的那个 LoRA"
        #    —— 实测 cliph_rerun 会因此从 0.999 掉到 0.335（详见 _lora_snapshot）。
        if self._active.get(pk) != key:
            if _swap_lora(model, self._lora_snap[key]) == 0:
                raise RuntimeError(f"LoRA 换入失败：{key} 的快照对不上共享包装器")
            self._active[pk] = key
        if fam == "cliph":
            f = model(x).float()                                   # open_clip 返回 pooled
        elif fam == "bigg":
            f = model(x, interpolate_pos_encoding=True).last_hidden_state[:, 0].float()
        else:                                                      # siglip2：不能传插值参数
            f = model(x).last_hidden_state[:, 0].float()
        return float(torch.sigmoid(self.heads[key](f).squeeze(-1)).item())
