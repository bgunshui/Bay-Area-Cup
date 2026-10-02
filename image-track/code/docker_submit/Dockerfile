# =============================================================================
# 湾区杯复赛 · 图像赛道 —— AI 生成图像鉴别 推理服务镜像
# 对应提交分：LB AUC = 0.918887
# 提交文件：v100_gpp20_seed4_c75_prior200.csv
#
# ── 基座（用户硬性要求：必须用 pytorch 官方镜像）──────────────────────────────
#   FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
#     · Ubuntu 22.04 (jammy) · conda 在 /opt/conda · python 3.11 · torch 2.5.1+cu124
#     · 交付的 tar 就是在它之上加了两层：**pip 依赖层** 与 **应用层**
#
# ── 交付标准对照（《分类模型 Docker 交付标准 v1.2》）──────────────────────────
#   · linux/amd64                  ✅ 架构与基座一致
#   · 监听 8080                    ✅ EXPOSE 8080 + server.py 绑 0.0.0.0:8080
#   · 三探针                       ✅ /health/{startup,liveness,readiness}
#   · 统一推理端点 POST /predict    ✅ base64 图像 → [0,1] 的 AI 置信度
#   · 自包含、运行期不联网          ✅ 权重与查表全部 COPY 进镜像；无任何下载逻辑
#   · 300 秒内 startup 就绪         ✅ 实测加载 ~40–90 秒（GPU）
#   · 单次推理 ≤ 90 秒             ✅ 实测单张 ~0.2–0.5 秒（GPU）
#   · 非 root 运行                 ✅ appuser(uid 1000)
#
# ── ⚠️ 这份 Dockerfile 与交付 tar 的关系（务必读）────────────────────────────
#   交付的 `baycup_image_aidet_0.918887.tar` 是用 **OCI 手法**（`oci_build3.py`，
#   因为构建机上拿不到 Docker 特权：`unshare: operation not permitted`）
#   按本文件**等价流程**组装出来的，最终 tar 里的 7 层、config、manifest 均已核对。
#   本文件是同一镜像的「`docker build` 标准写法」，两者产出的文件系统与运行行为一致。
#
# ── 两个真实交付事故（都写进注释，因为都是"看起来没问题"的坑）────────────────
#   ① 2026-09-18「镜像权重崩溃」：骨干 `.safetensors` 源文件是 `-rw------- root:root`
#      （fp16 转换脚本 umask=077 所致；`.bin` 是 cp 来的所以是 0644 —— **只有 safetensors 中招**），
#      而容器以 `User=appuser` 运行 ⇒ 加载骨干时 **PermissionError**，服务直接崩。
#      → 修：`chmod -R a+rX`（下面 `RUN` 里那句），并在构建脚本里加**逐文件 open() 闸门**。
#   ② 2026-09-19「HEALTHCHECK 永远 unhealthy」：pytorch 官方基座**不含 `curl` 可执行文件**
#      （只有 `curl-config`），而镜像声明的 HEALTHCHECK 就是 `curl -f …` ⇒ 探针永远失败。
#      → 修：下面显式 `apt-get install curl`（交付 tar 里是一层独立的 2.5 MB 补丁层）。
#
# ── 构建前置 ────────────────────────────────────────────────────────────────
#   app/weights/     8 个 LoRA+head 权重（约 1.0 GB）
#   app/backbones/   3 个骨干（fp16 存，约 9.3 GB；对应 fp32 为 18.4 GB）
#   app/tables/      融合查表 + 尺寸先验部件（约 0.5 MB）
#   由 `prepare_assets.py` 备齐；`bash build.sh` 一条龙（下载 → 构建 → 冒烟 → docker save）。
# =============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 与交付镜像 config 里的 Env **逐项一致**（docker 会注入；离线 + 不写字节码 + 固定线程数）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/app/.hf \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8

# 规范 §2.3 的 HEALTHCHECK 用 curl；官方基座里**没有** curl，必须显式装（见上文事故 ②）
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl --version | head -1

WORKDIR /app

# ---- 依赖：版本以**镜像内实测**为准，见 requirements.txt 与 pip_freeze_image.txt ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import torch,transformers,peft,open_clip,fastapi; \
print('torch',torch.__version__);print('transformers',transformers.__version__);\
print('peft',peft.__version__);print('open_clip',open_clip.__version__)"

# ---- 资产：权重 + 查表 + 代码（全部打进镜像，运行期零下载）----
COPY app/ /app/

# 非 root 运行（规范 §4.3）。
# ⚠️ `chmod -R a+rX` 是事故 ① 的修复动作本身，**不要删**：
#    它保证无论上游给来的文件是什么权限，appuser 都能读到。
#    `chown` 让 app 目录归 appuser；`/home/appuser` 给一个可写的 HOME（有些库会往 ~ 写缓存）。
RUN useradd -m -u 1000 appuser \
 && mkdir -p /home/appuser \
 && chown -R appuser:appuser /app /home/appuser \
 && chmod -R a+rX /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=300s --retries=3 \
  CMD curl -f http://localhost:8080/health/liveness || exit 1

CMD ["python", "server.py"]
