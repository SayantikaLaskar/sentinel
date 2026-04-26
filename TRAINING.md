# Training Guide

This repo is ready for **LLM-only** GRPO training. Real runs require a CUDA GPU (Google Colab T4 minimum, A100/L40S recommended).

## Recommended Setup

Use:
- `A100-80GB` or `L40S-48GB`
- `Qwen/Qwen2.5-7B-Instruct`
- `--no-4bit`

That is the safest path for this codebase right now. It avoids the bitsandbytes CUDA-runtime mismatch that can show up with 4-bit hosted runs.

For a judge-friendly rerunnable notebook, use:
- `sentinel_colab_training.ipynb`

## Agent Priority

Train in this order:

1. `holmes`
2. `forge`
3. `hermes`
4. `oracle`
5. `argus`

What each one does:
- `holmes`: diagnosis and root-cause investigation
- `forge`: remediation and service recovery
- `hermes`: deployment safety and rollout control
- `oracle`: escalation, closure, and scenario control
- `argus`: monitoring support and evidence gathering

Why `holmes` and `forge` first:
- they dominate the benchmark outcome
- `holmes` drives diagnosis quality
- `forge` drives MTTR, recovery, and blast-radius reduction

## Local Or Other GPU Machine

Install:

```bash
git clone https://github.com/sayantikalaskar/sentinel.git
cd sentinel
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install unsloth trl datasets
```

Train:

```bash
python train.py --agent holmes --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 500 --batch-size 2
python train.py --agent forge --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 500 --batch-size 2
python train.py --agent hermes --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 300 --batch-size 2
python train.py --agent oracle --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 300 --batch-size 2
python train.py --agent argus --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 300 --batch-size 2
```

Resume:

```bash
python train.py --agent holmes --model Qwen/Qwen2.5-7B-Instruct --no-4bit --resume --checkpoint-dir checkpoints
python train.py --agent forge --model Qwen/Qwen2.5-7B-Instruct --no-4bit --resume --checkpoint-dir checkpoints
```

Evaluate:

```bash
python train.py --agent holmes --model Qwen/Qwen2.5-7B-Instruct --no-4bit --eval-only --eval-episodes 20
python train.py --agent forge --model Qwen/Qwen2.5-7B-Instruct --no-4bit --eval-only --eval-episodes 20
```

## Files To Bring Back

Copy these back after training:
- `checkpoints/holmes/`
- `checkpoints/forge/`
- `checkpoints/hermes/`
- `checkpoints/oracle/`
- `checkpoints/argus/`
- per-agent training logs if you split them

For cleaner logs:

```bash
python train.py --agent holmes --log-file holmes_training_log.jsonl
python train.py --agent forge --log-file forge_training_log.jsonl
```
