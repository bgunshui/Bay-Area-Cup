FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    LOAD_8BIT=1 \
    FOLDS=5 \
    MAX_LEN=512

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 模型权重与 adapters(构建前需放好,禁止运行期下载)
COPY model/ /app/model/
COPY adapters/ /app/adapters/
COPY server.py /app/server.py

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=300s --retries=3 \
  CMD curl -f http://localhost:8080/health/liveness || exit 1

CMD ["python", "server.py"]
