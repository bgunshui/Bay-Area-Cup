cd /hy-tmp/build
python3 - <<'PY'
import sys, warnings, torch
warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from PIL import Image
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, "/hy-tmp/build/buildkit/docker_submit/app")
from models import COMPONENTS, STD, _tfm, load_lora_into, detect_r
from transformers import CLIPVisionModel
from peft import LoraConfig, get_peft_model

APP = Path("/hy-tmp/build/buildkit/docker_submit/app")
BB  = Path("/hy-tmp/build/backbones")
REF = Path("/hy-tmp/verify/ref")
KEY = "img_bigg_s384_retrain"
fam, wfile, size, mean = COMPONENTS[KEY]
tfm = _tfm(size, mean, STD[KEY])

sc = pd.read_csv("/hy-tmp/verify/host_scores.csv", dtype={"id": str})
ids = sc["id"].tolist()
cands = [c for c in Path("/hy-tmp/dl/fusai").rglob("image_test") if c.is_dir()]
img_root = max(cands, key=lambda c: len(list(c.glob("*.jpg"))))
imgs = {}
for i in ids:
    p = img_root / i
    if p.exists():
        im = Image.open(p); im.load(); imgs[i] = im.convert("RGB")
ids = [i for i in ids if i in imgs]
print("图片 %d 张" % len(ids))

def build(dtype):
    v = CLIPVisionModel.from_pretrained(str(BB/"clip_bigg"), torch_dtype=dtype)
    r = detect_r(torch.load(APP/"weights"/wfile, map_location="cpu", weights_only=False)["lora"])
    m = get_peft_model(v, LoraConfig(r=r, lora_alpha=2*r, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"])).to("cuda").eval()
    hs, n = load_lora_into(m, APP/"weights"/wfile, None)
    head = torch.nn.Linear(1664, 1).to("cuda"); head.load_state_dict(hs); head.eval()
    return m, head

def run(m, head, dt, autocast_dtype=None):
    out = []
    with torch.inference_mode():
        for i in ids:
            x = tfm(imgs[i]).unsqueeze(0).to("cuda").to(dt)
            if autocast_dtype is None:
                f = m(x, interpolate_pos_encoding=True).last_hidden_state[:, 0].float()
            else:
                with torch.autocast("cuda", dtype=autocast_dtype):
                    f = m(x, interpolate_pos_encoding=True).last_hidden_state[:, 0].float()
            out.append(float(torch.sigmoid(head(f).squeeze(-1)).item()))
    return np.array(out)

frozen = pd.read_csv(REF/"img_bigg_s384_retrain.csv", encoding="utf-8-sig",
                     dtype={"id": str}).set_index("id")["score"].reindex(ids).to_numpy()
print("冻结           mean=%.4f std=%.4f" % (frozen.mean(), frozen.std()))

for tag, dt, ac in (("fp32 + bf16 autocast（训练侧）", torch.float32, torch.bfloat16),
                    ("fp16（服务侧，当前）", torch.float16, None)):
    m, head = build(dt)
    p = run(m, head, dt, ac)
    print("%-30s mean=%.4f std=%.4f   spearman=%.6f pearson=%.6f max|Δ|=%.4f"
          % (tag, p.mean(), p.std(), spearmanr(p, frozen).statistic,
             np.corrcoef(p, frozen)[0,1], np.abs(p-frozen).max()))
    del m, head
    torch.cuda.empty_cache()
PY
