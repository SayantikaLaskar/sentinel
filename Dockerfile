# SENTINEL — Multi-Agent Incident Response RL Environment
# Base: Python 3.11 slim
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# ── Model weights download instructions ──────────────────────────────────────
# To download Llama-3-8B-Instruct weights for HOLMES and FORGE training:
#
#   Option 1 — Unsloth (recommended, 4-bit quantized):
#     pip install unsloth
#     python -c "from unsloth import FastLanguageModel; \
#       FastLanguageModel.from_pretrained('unsloth/Meta-Llama-3-8B-Instruct', \
#       max_seq_length=4096, load_in_4bit=True)"
#
#   Option 2 — HuggingFace Hub (full precision, requires ~16GB VRAM):
#     pip install huggingface_hub
#     python -c "from huggingface_hub import snapshot_download; \
#       snapshot_download('meta-llama/Meta-Llama-3-8B-Instruct', \
#       local_dir='/app/models/llama3-8b')"
#
#   Option 3 — Pre-download at build time (add to Dockerfile):
#     ARG HF_TOKEN
#     RUN pip install huggingface_hub && \
#         python -c "from huggingface_hub import snapshot_download; \
#         snapshot_download('unsloth/Meta-Llama-3-8B-Instruct', \
#         local_dir='/app/models/llama3-8b', token='${HF_TOKEN}')"
#
# Note: The SENTINEL environment runs without model weights in simulation mode.
# Weights are only required for GRPO fine-tuning of HOLMES and FORGE agents.

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SENTINEL_DEBUG=false

# ── Expose ports ──────────────────────────────────────────────────────────────
# FastAPI server
# Gradio dashboard
EXPOSE 7860

# ── Default command: start FastAPI server ─────────────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
