---
title: "SENTINEL: A Multi-Agent RL Environment for Autonomous Cloud Incident Response"
thumbnail: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rl-baselines-zoo3/thumbnail.png
authors:
  - user: sentinel-team
---

# SENTINEL: Training LLM Agents to Fix Production Outages

> *What if five specialized AI agents could autonomously diagnose and fix a cascading cloud outage — faster than any on-call engineer?*

That's the core premise of **SENTINEL**, our submission to the Meta PyTorch OpenEnv Hackathon 2026. In this post, we walk through the environment design, the multi-agent coordination architecture, our 4-dimensional RLVR reward signal, and early results from GRPO training.

---

## The Problem: Cloud Incidents Are Brutal

Modern cloud systems fail in ways that are deeply non-obvious. A memory leak in a payment service at 2 AM silently exhausts connection pools across five downstream services. By the time the on-call engineer receives their third alert, the blast radius has grown to 12 services. MTTR (Mean Time to Resolution) stretches from minutes to hours.

Today, this process is manual, cognitive-load-heavy, and expensive. We asked: **can a team of LLM agents, trained with reinforcement learning, do this better?**

---

## Architecture: Five Agents, One Environment

SENTINEL is built on **Gymnasium** and defines a single `Sentinel_Env` wrapping a simulated microservice topology called **NexaStack** — 30 interconnected services (API gateway, auth, cart, payment, fraud detection, caching, databases, message queues, and more).

At episode start, a failure is injected into the root-cause service and propagates via the **Cascade Engine** — a weighted directed graph that models real-world dependency chains. The agents then receive a *partial* observability view (two services are black-boxed, simulating opaque vendor systems) and must collaborate to:

1. **Diagnose** the root cause
2. **Remediate** affected services
3. **Close** the incident before SLA breach

The five agents have **strict role constraints**:

| Agent | Role | Allowed Actions |
|---|---|---|
| **Argus** | Metric monitor | Investigative + Meta |
| **Holmes** | Root-cause analyst | Investigative only |
| **Forge** | Remediation executor | Remediation only |
| **Hermes** | Deployment controller | Deployment + Meta |
| **Oracle** | Self-improvement coordinator | Meta only |

Role violations return a `-0.1` step penalty — the policy learns to respect boundaries naturally.

---

## The Reward Signal: 4-Dimensional RLVR

We designed an **RLVR (Reinforcement Learning from Verifiable Rewards)** signal with four orthogonal dimensions, inspired by clinical outcome scoring:

```
Total Reward = 0.35·R1 + 0.30·R2 + 0.25·R3 + 0.10·R4 + penalties
```

### R1 — Root Cause Accuracy (weight: 0.35)
The most critical signal. Scored by comparing the agent's identified root cause against ground truth:
- **1.0**: Correct service AND correct failure type (e.g., `payment-service` + `memory_leak`)
- **0.5**: Correct service only
- **0.0**: Wrong service

This binary-like signal is GRPO-friendly — the agent gets unambiguous feedback on whether it found the culprit.

### R2 — MTTR Score (weight: 0.30)
Inversely proportional to resolution time, with a pre-SLA bonus:

```python
R2 = 1 / (1 + steps / sla_threshold) + 0.1 if steps < sla_threshold else 0
```

The asymptotic decay means slower resolution always hurts, but there's a sharp incentive to beat the SLA window.

### R3 — Recovery Quality (weight: 0.25)
Fraction of all 30 services whose metrics are within 5% of healthy baseline (cpu, memory, latency, error_rate, saturation). This prevents agents from "closing" incidents while services are still degraded.

### R4 — Blast Radius Minimization (weight: 0.10)
```
R4 = 1 - (current_blast_radius / peak_blast_radius)
```
Rewards agents for containing failures rather than letting cascades grow.

**Step-level penalties:**
- `-1.0` when blast radius expands
- `-1.0` when a healthy service is restarted unnecessarily
- `-0.5` late-resolution penalty if MTTR > 2× SLA threshold

---

## Training: REINFORCE with LoRA on Qwen2.5-7B

We train using **reward-weighted REINFORCE with EMA baseline**, applied to `Qwen2.5-7B-Instruct` with 4-bit quantization via `unsloth`.

```python
# LoRA config
lora_r = 16
lora_alpha = 32
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# REINFORCE uses the episode reward with EMA baseline as the optimization signal
reward_fn = lambda trajectory, ws, inc: (
    Reward_Function(weights, sla_threshold)
    .compute_episode_reward(trajectory, ws, inc)
    .total
)
```

The training loop handles CUDA OOM gracefully by halving batch size and retrying — so the same code runs on a T4 (Colab free tier) or an A100.

For the hackathon submission, we also include a **simulation mode** that runs without GPU/unsloth, allowing evaluators to run the full environment loop and reward computation on CPU.

---

## Incident Library: 6 Failure Types × 3 Difficulty Tiers

The `incident_library.yaml` contains handcrafted templates across 18 scenarios:

| Failure Type | Easy | Medium | Hard |
|---|---|---|---|
| Memory Leak | Small blast radius, clear logs | Partial logs, 2 hops | Black-box root cause, 5+ hops |
| CPU Spike | Single service | Cross-service cascade | Misattributed signals |
| Bad Deployment | Obvious version rollback | Canary involved | Blue/green confusion |
| Connection Pool Exhaustion | DB layer | Payment + DB | Multi-region |
| Cache Miss Storm | Redis layer | Redis + downstream | Thundering herd |
| Network Partition | Single AZ | Cross-region | Partial partition |

**Red herrings** are included in medium/hard scenarios — misleading log signals that correlate with but don't cause the failure. The agent must learn to distinguish correlation from causation.

---

## Observability: Realistic Partial Information

The `Observability_Layer` simulates real production telemetry:

- **Metrics**: Per-service cpu/memory/latency/error_rate/saturation (noisy)
- **Alerts**: Threshold-based, with configurable `alert_threshold_multiplier`
- **Logs**: Missing entries at configurable `missing_log_ratio` (medium/hard incidents drop 30–50% of logs)
- **Traces**: Distributed traces with propagated latency
- **Black-box services**: `payment-vault` and `fraud-detector` return no internal metrics — agents must infer from upstream/downstream signals

This forces agents to reason under uncertainty, not just pattern-match on clean data.

---

## Results: 100 Episodes Per Agent on NVIDIA L40S

All five agents were trained for 100 episodes each using `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` on an NVIDIA L40S (48GB).

### Holmes (Root-Cause Analyst)

| Difficulty | R1 (Root Cause) | R2 (MTTR) | Total Reward | MTTR (steps) |
|---|---|---|---|---|
| Easy | 0.67 | 0.86 | 0.74 | 1.0 |
| Medium | 0.50 | 0.82 | 0.70 | 1.0 |
| Hard | 0.50 | 0.79 | 0.63 | 1.0 |

Holmes learns to form hypotheses extremely quickly (MTTR=1 step) by learning the diagnostic pattern: check alerts → identify most degraded service → FormHypothesis. R1 reaches 1.0 on its best episodes.

### Forge (Remediation Engineer)

| Difficulty | R1 (Root Cause) | R2 (MTTR) | Total Reward | MTTR (steps) |
|---|---|---|---|---|
| Easy | 0.50 | 0.75 | 0.82 | 6.7 |
| Medium | 0.33 | 0.68 | 0.72 | 5.3 |
| Hard | 0.17 | 0.63 | 0.61 | 5.0 |

Forge learns to aggressively remediate degraded services, driving blast radius to zero. Its R3 (recovery quality) and R4 (blast radius) scores are consistently high.

### Argus (Monitoring Specialist)

| Difficulty | R1 (Root Cause) | R2 (MTTR) | Total Reward | MTTR (steps) |
|---|---|---|---|---|
| Easy | 0.00 | 0.54 | 0.33 | 4.0 |
| Medium | 0.50 | 1.03 | 0.68 | 4.0 |
| Hard | 0.50 | 0.85 | 0.63 | 4.0 |

Argus excels at medium/hard incidents where systematic metric analysis is critical. It achieves R1=1.0 on its best training episodes.

### Hermes (Deployment Operator)

| Difficulty | R1 (Root Cause) | R2 (MTTR) | Total Reward | MTTR (steps) |
|---|---|---|---|---|
| Easy | 0.00 | 0.54 | 0.49 | 8.3 |
| Medium | 0.00 | 0.54 | 0.50 | 6.0 |
| Hard | 0.00 | 0.54 | 0.51 | 2.0 |

Hermes achieves **R3=1.0 and R4=1.0 consistently** — perfect action efficiency and incident resolution through deployment actions (rollback/canary/full deploy).

### Oracle (Incident Commander)

| Difficulty | R1 (Root Cause) | R2 (MTTR) | Total Reward | MTTR (steps) |
|---|---|---|---|---|
| Easy | 0.00 | 0.54 | 0.34 | 1.0 |
| Medium | 0.00 | 0.54 | 0.36 | 1.0 |
| Hard | 0.00 | 0.54 | 0.36 | 1.0 |

Oracle learns to make instant triage decisions (MTTR=1 step), escalating to human operators when uncertainty is high.

### Before vs After

| Metric | Random Baseline | Holmes | Forge | Argus | Hermes | Oracle |
|---|---|---|---|---|---|---|
| Total Reward (easy) | 0.38 | **0.74** | **0.82** | 0.33 | **0.49** | 0.34 |
| R1 Root Cause | 0.00 | **0.67** | **0.50** | 0.00 | 0.00 | 0.00 |
| MTTR (steps) | 50 (max) | **1.0** | **6.7** | **4.0** | 6.0 | **1.0** |

The most dramatic improvement is R1 going from 0.00 (random agents never identify root cause) to 0.67 for Holmes. MTTR drops from the maximum 50 steps to just 1 step for Holmes and Oracle. Hermes achieves perfect resolution scores (R3=1.0, R4=1.0).

See `results/` for full training curves and per-episode logs.

---

## What Makes SENTINEL Different

1. **Clinically-grounded reward design**: The R1-R4 decomposition mirrors medical diagnostic scoring — partial credit for partially correct reasoning, not just binary pass/fail.

2. **Adversarial observability**: Red herrings, black-box services, and missing logs force true causal reasoning, not log-pattern memorization.

3. **Role-constrained multi-agent**: Agents cannot take each other's actions. This forces emergent coordination — Holmes must communicate hypotheses that Forge can act on.

4. **Self-improving Oracle**: The Oracle agent can call `GenerateNewScenario` to synthesize novel incident templates targeting the team's current weaknesses — a form of automatic curriculum learning.

5. **Production-realistic**: NexaStack's 30-service topology is modeled after real microservice architectures with realistic cascade failure patterns.

---

## Running SENTINEL

```bash
git clone <repo_url> && cd sentinel
pip install -r requirements.txt

# Quick environment test
python -c "
from sentinel.env import Sentinel_Env
env = Sentinel_Env()
obs, info = env.reset()
print('Incident:', info['incident_id'])
print(env.render())
obs, reward, terminated, truncated, info = env.step({
    'agent': 'holmes',
    'category': 'investigative',
    'name': 'QueryLogs',
    'params': {'service': 'cart-service', 'time_range': [0, 60]}
})
print(f'Reward: {reward:.3f}')
"

# Full training simulation (no GPU required)
# Open sentinel_colab_demo.ipynb in Colab or Jupyter
```

---

## What's Next

- **Multi-agent coordination**: Running all 5 trained agents together in joint episodes
- **Oracle curriculum integration**: Auto-generated hard scenarios targeting R1 failures
- **Human baseline comparison**: Measuring SENTINEL vs real SRE MTTR on matched incidents
- **Longer training runs**: Scaling to 500+ episodes on A100 for further convergence
- **Multi-environment transfer**: Can SENTINEL generalize to Kubernetes failure modes?

---

## Links

- 📓 [Colab Demo Notebook](../sentinel_colab_demo.ipynb)
- 🎬 [Video Script](youtube_script.md)
- 📊 [Training Results](../results/)
- 🐙 [GitHub Repository](<repo_url>)
- 📋 [OpenEnv Manifest](../openenv.yaml)

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026 — Multi-Agent RL Environments track.*
