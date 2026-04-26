---
title: SENTINEL
emoji: "🛡️"
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# 🛡️ SENTINEL — Multi-Agent Incident Response Environment

> **Train LLM agents to diagnose and resolve production outages in a 30-service microservice platform.**

SENTINEL is a Gymnasium-compatible RL environment where AI agents navigate realistic cloud incidents — cascading failures, partial observability, misleading alerts, and a ticking SLA clock. Built for the **Meta PyTorch OpenEnv Hackathon 2026**.

[![Live Space](https://img.shields.io/badge/🤗_HF_Space-Live-blue)](https://harry1911-sentinel.hf.space/dashboard/)
[![API Docs](https://img.shields.io/badge/API-Docs-green)](https://harry1911-sentinel.hf.space/docs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![SENTINEL Dashboard](assets/dashboard_overview.jpg)

---

## Submission Assets

| Deliverable | Link |
|-------------|------|
| GitHub Repository | [github.com/sayantikalaskar/sentinel](https://github.com/sayantikalaskar/sentinel) |
| Hugging Face Space | [huggingface.co/spaces/harry1911/sentinel](https://huggingface.co/spaces/harry1911/sentinel) |
| Live Dashboard | [harry1911-sentinel.hf.space/dashboard](https://harry1911-sentinel.hf.space/dashboard/) |
| Health Endpoint | [harry1911-sentinel.hf.space/health](https://harry1911-sentinel.hf.space/health) |
| API Docs | [harry1911-sentinel.hf.space/docs](https://harry1911-sentinel.hf.space/docs) |
| OpenEnv Manifest | [`openenv.yaml`](openenv.yaml) |
| Training Notebook | [`sentinel_colab_training.ipynb`](sentinel_colab_training.ipynb) |
| Training Scripts | [`train.py`](train.py), [`retrain.py`](retrain.py) |
| Blog Write-up | [`Blog.MD`](Blog.MD) |
| Training Curves | [`results/`](results/) |
| TensorBoard Logs | [`results/runs/`](results/runs/) |

---

## Quick Start

```bash
git clone https://github.com/sayantikalaskar/sentinel.git
cd sentinel
pip install -r requirements.txt

# Run a single episode
python -c "
from sentinel.env import Sentinel_Env
env = Sentinel_Env()
obs, info = env.reset(seed=42)
print('Incident:', info['incident_id'])
print('Alerts:', len(obs['active_alerts']))
"
```

---

## What SENTINEL Simulates

Each episode models a **cloud outage** across a 30-service platform called NexaStack:

- A failure is injected into the service dependency graph
- Cascading damage propagates through downstream services
- The agent observes partial telemetry and must act under uncertainty
- Every action changes the system state — wrong actions make things worse

### Observation Space (7 channels)

| Channel | What It Provides |
|---------|-----------------|
| `metrics_snapshot` | CPU, error rate, latency per service |
| `active_alerts` | Currently firing alerts (may be misleading) |
| `causal_graph_snapshot` | Partial view of service dependencies |
| `recent_logs` | Log entries from visible services |
| `active_traces` | Distributed traces showing request flow |
| `incident_context` | High-level incident metadata |
| `sla_state` | Time remaining before SLA breach |

![Service Health Grid](assets/service_health.jpg)
*A live incident: `cart-service` is the root cause (ROOT badge), but failure has cascaded to 8+ downstream services.*

---

## Multi-Agent Architecture

Five specialized agent roles mirror real incident response teams:

| Agent | Role | Allowed Actions |
|-------|------|----------------|
| 🔍 **Holmes** | Root-cause analyst | QueryLogs, QueryMetrics, AnalyzeTraces, FormHypothesis |
| 📊 **Argus** | Monitoring support | Investigative + Meta (correlate, check SLA) |
| 🔧 **Forge** | Remediation executor | RestartService, ScaleUp, Rollback |
| 📡 **Hermes** | Deployment controller | Deploy, Rollback, ConfigChange |
| 🚨 **Oracle** | Incident command | EscalateToHuman, CloseIncident |

Each role is **constrained** — Holmes cannot restart services, Forge cannot query logs. This forces specialized reasoning.

![Agent Actions](assets/agent_action.jpg)

---

## Reward Design

A weighted, decomposable reward that separates diagnosis from remediation:

| Component | Weight | Measures |
|-----------|--------|----------|
| **R1** — Root Cause Accuracy | 35% | Did the agent find the *actual* root cause? |
| **R2** — MTTR Efficiency | 30% | How fast was the resolution? |
| **R3** — Recovery Quality | 25% | What fraction of services recovered? |
| **R4** — Blast Radius Control | 10% | Was damage contained or expanded? |

Plus **shaping penalties** for restarting healthy services or expanding blast radius.

---

## OpenEnv Integration

```yaml
# openenv.yaml
spec_version: 1
name: sentinel
type: space
runtime: fastapi
app: server.app:app
port: 7860
```

The adapter exposes the standard OpenEnv interface:

- `reset(seed, options)` → Initialize a new incident
- `step(action)` → Execute an agent action
- `state` → Inspect current adapter state

**Key files:**
- Environment wrapper: [`server/sentinel_environment.py`](server/sentinel_environment.py)
- Runtime entrypoint: [`server/app.py`](server/app.py)
- Core Gym environment: [`sentinel/env.py`](sentinel/env.py)
- Manifest: [`openenv.yaml`](openenv.yaml)

---

## Live Validation

The public Hugging Face Space provides a validation dashboard:

![Smoke Test](assets/smoke_test.jpg)

- **Dashboard UI**: [harry1911-sentinel.hf.space/dashboard](https://harry1911-sentinel.hf.space/dashboard/)
- **Health check**: [harry1911-sentinel.hf.space/health](https://harry1911-sentinel.hf.space/health)
- **API docs**: [harry1911-sentinel.hf.space/docs](https://harry1911-sentinel.hf.space/docs)

The dashboard can reset episodes, execute agent actions, run a full smoke test (`reset` → `step` → `state` → schema validation), and inspect live observation/state JSON.

---

## Training

### Stack

- **Model:** `Qwen2.5-7B-Instruct` (4-bit via Unsloth)
- **Method:** GRPO with LoRA adapters (r=16, α=32)
- **Hardware:** NVIDIA L40S (48GB)
- **Tracking:** TensorBoard (enabled by default)

### Train All 5 Agents

```bash
# Recommended: trains all agents sequentially with CUDA isolation
python retrain.py

# Or train individually:
python train.py --agent holmes --episodes 100 --batch-size 2
python train.py --agent forge --episodes 100 --batch-size 2
python train.py --agent argus --episodes 100 --batch-size 2
python train.py --agent hermes --episodes 100 --batch-size 2
python train.py --agent oracle --episodes 100 --batch-size 2
```

### Latest Results (5 agents × 100 episodes)

| Agent | Avg Reward | Best Reward | R1 (root cause) | Avg MTTR | Time |
|-------|-----------|-------------|-----------------|----------|------|
| **Holmes** | **0.771** | **0.899** | **0.75** | 4.3 steps | 709s |
| **Forge** | **0.760** | **0.849** | 0.40 | 5.8 steps | 634s |
| **Argus** | **0.743** | **0.899** | **0.65** | 4.2 steps | 707s |
| Hermes | 0.497 | 0.512 | 0.00 | 7.2 steps | 621s |
| Oracle | 0.366 | 0.404 | 0.00 | 1.0 steps | 166s |

**Holmes achieves 100% root-cause accuracy on easy incidents** and 67% on medium/hard.

---

## Training Curves

### Comparison Plots

![All Agents Comparison](results/comparison_all_agents.png)

![Loss Comparison](results/comparison_loss.png)

<details>
<summary><strong>Per-Agent Curves</strong> (click to expand)</summary>

#### Holmes
![Holmes Training](results/holmes_training_curves.png)
![Holmes Loss](results/holmes_loss_curve.png)

#### Forge
![Forge Training](results/forge_training_curves.png)
![Forge Loss](results/forge_loss_curve.png)

#### Argus
![Argus Training](results/argus_training_curves.png)
![Argus Loss](results/argus_loss_curve.png)

#### Hermes
![Hermes Training](results/hermes_training_curves.png)
![Hermes Loss](results/hermes_loss_curve.png)

#### Oracle
![Oracle Training](results/oracle_training_curves.png)
![Oracle Loss](results/oracle_loss_curve.png)

</details>

---

## Experiment Tracking

TensorBoard is enabled by default. Every training run logs:

- **Per-episode scalars:** R1, R2, R3, R4, total reward, MTTR, loss
- **Hyperparameters:** agent, model, LoRA config, batch size
- **Final summary:** hparam table with aggregate metrics

```bash
pip install tensorboard
tensorboard --logdir results/runs/
```

Event files committed in [`results/runs/`](results/runs/). Toggle via `experiment_tracking=True/False` in `TrainingConfig`.

---

## Project Structure

```
sentinel/
├── sentinel/              # Core Gymnasium environment
│   ├── env.py             # reset(), step(), render()
│   ├── cascade_engine.py  # Failure propagation
│   ├── math_engine.py     # Reward computation (R1-R4)
│   ├── config.py          # YAML-driven config
│   └── agents/            # Holmes, Forge, Argus, Hermes, Oracle
├── sentinel/training/     # GRPO training pipeline
│   ├── pipeline.py        # Training loop + TensorBoard
│   ├── llm_agent.py       # LLM action generation
│   ├── prompt_builder.py  # Observation → prompt
│   └── evaluate.py        # Multi-tier evaluation
├── server/                # OpenEnv adapter
│   ├── app.py             # FastAPI + Gradio mount
│   └── sentinel_environment.py
├── demo/app.py            # Gradio dashboard
├── assets/                # Dashboard screenshots
├── results/               # Training artifacts + TensorBoard runs
├── train.py               # Training entrypoint
├── retrain.py             # Full 5-agent retraining
├── openenv.yaml           # OpenEnv manifest
├── Dockerfile             # HF Space container
├── Blog.MD                # Detailed write-up
└── sentinel_colab_training.ipynb  # Colab notebook
```

---

## Validation Checklist

- [x] Public, logged-out, cloneable Hugging Face Space
- [x] Parseable [`openenv.yaml`](openenv.yaml) with OpenEnv adapter (`reset`, `step`, `state`)
- [x] Committed reward and loss plots for all 5 agents
- [x] Runnable training notebook and scripts
- [x] TensorBoard experiment tracking enabled
- [x] README links every required deliverable
- [x] Blog write-up with architecture, results, and screenshots

---

## Hackathon Category

**Primary:** World Modeling
**Secondary:** Long-Horizon Planning

## Authors

**Harsh Shukla** and **Sayantika Laskar**

Built for the Meta PyTorch OpenEnv Hackathon 2026.
Trained on NVIDIA L40S. Powered by Qwen2.5-7B + Unsloth + GRPO.

## License

MIT
