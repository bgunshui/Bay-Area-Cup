#!/usr/bin/env python3
"""infer_gpp.py: 用已保存的 best.pt 对测试集 20000 张推理，输出探针 CSV。"""
import glob
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image as PImage
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATA = Path("/hy-tmp/baycup/data")
RUN = Path("/hy-tmp/baycup/runs/gpp")
MODEL_DIR = Path("/hy-tmp/baycup/models/siglip2_384")
SIZE = int(os.environ.get("S", "384"))
BS = int(os.environ.get("BS", "16"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
VAL_TFM = transforms.Compose([
    transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(*NORM),
])


class DS(Dataset):
    def __init__(self, root, names):
        self.root, self.names = root, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        try:
            im = PImage.open(self.root / self.names[i]).convert("RGB")
        except Exception:
            im = PImage.new("RGB", (SIZE, SIZE))
        return VAL_TFM(im), 0


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import SiglipVisionModel

    vision = SiglipVisionModel.from_pretrained(str(MODEL_DIR), local_files_only=True)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
    vision = get_peft_model(vision, lcfg).to(DEVICE)
    head = torch.nn.Linear(vision.config.hidden_size, 1).to(DEVICE)
    state = torch.load(RUN / "best.pt", map_location="cpu")
    vision.load_state_dict(state["lora"])
    head.load_state_dict(state["head"])
    print("loaded best.pt", flush=True)

    replay_csv = DATA / "replay" / "image_test.csv"
    replay = pd.read_csv(replay_csv, encoding="utf-8-sig", dtype={"id": str})
    root = replay_csv.parent / "data" / "image_test"
    ds = DS(root, replay.id.astype(str).tolist())
    dl = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=10)
    vision.eval(); head.eval()
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, (x, _) in enumerate(dl):
            f = vision(x.to(DEVICE, non_blocking=True)).last_hidden_state[:, 0].float()
            out.append(head(f).squeeze(-1).float().cpu().numpy())
            if i % 200 == 0:
                print("step", i, "/", len(dl), flush=True)
    rs = np.concatenate(out)
    dst = RUN / "img_gpp_siglip2.csv"
    pd.DataFrame({"id": replay.id.astype(str), "score": 1.0 / (1.0 + np.exp(-rs))}).to_csv(dst, index=False)
    print("DONE", dst, len(rs), flush=True)


if __name__ == "__main__":
    main()
