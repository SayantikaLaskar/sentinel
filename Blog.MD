# SENTINEL: Training LLM Agents to Diagnose and Resolve Cloud Outages

## TL;DR

SENTINEL is an OpenEnv-compatible reinforcement learning environment for autonomous cloud incident response. Instead of treating operations as a static QA problem, it simulates a live microservice platform where failures propagate, observability is incomplete, and every action changes the future state of the system.

The benchmark asks a model to do what real on-call engineers do:

1. identify the actual root cause
2. avoid misleading telemetry
3. choose a safe remediation
4. reduce blast radius quickly
5. close the incident before the SLA window is gone

This repo includes:

- a Gym-style environment with `reset()` and `step()`
- an OpenEnv-compatible server adapter
- a runnable training path via [`train.py`](train.py) and [`sentinel_colab_training.ipynb`](sentinel_colab_training.ipynb)
- committed training plots in [`results/`](results/)

## The Problem

Most LLM benchmarks for operations are too static. They check whether a model can explain a log line or answer an SRE-flavored question, but real incidents are not single-turn reasoning tasks.

Real production outages are adversarial:

- the first alert is often wrong or incomplete
- the loudest signal may be a downstream symptom
- logs can be missing
- hidden services can break local reasoning
- a bad action can expand the blast radius and make the outage worse

SENTINEL is built around that gap. We wanted an environment where the model has to act under pressure, not just describe what it would do.

## The World: NexaStack

SENTINEL simulates **NexaStack**, a 30-service cloud application with realistic service dependencies. At reset time, the environment injects an incident into the dependency graph and lets that failure propagate through downstream services.

The observation space exposes:

- `metrics_snapshot`
- `active_alerts`
- `causal_graph_snapshot`
- `recent_logs`
- `active_traces`
- `incident_context`
- `sla_state`

The environment is intentionally only partially observable. Some services behave like real vendor or black-box infrastructure: the model has to infer what is happening indirectly from surrounding telemetry.

## Why This Benchmark Is Hard

The benchmark is hard in ways that matter for agentic RL:

- incidents are multi-step and stateful
- wrong actions have consequences
- evidence can be incomplete or misleading
- diagnosis and remediation are not the same skill
- speed matters, but blind speed is punished

That combination makes SENTINEL a better fit for long-horizon agent training than toy tool-use environments.

## The Multi-Agent Framing

We structured the action space around five specialized operational roles:

| Agent | Role | Allowed actions |
|---|---|---|
| `holmes` | root-cause analyst | investigative |
| `forge` | remediation executor | remediation |
| `hermes` | deployment controller | deployment, meta |
| `oracle` | incident command | meta |
| `argus` | monitoring support | investigative, meta |

This matters because incident response is not just about capability, it is also about coordination and constrained responsibility. A strong benchmark should reward correct behavior inside realistic operational boundaries, not just unconstrained action spam.

In practice, the strongest training targets are `holmes` and `forge`:

- `holmes` determines whether the system identifies the real root cause
- `forge` determines whether the system recovers quickly and safely

## Reward Design

The reward is not a single vague success score. It is a weighted, verifiable decomposition:

- `R1`: root-cause accuracy
- `R2`: MTTR efficiency
- `R3`: recovery quality
- `R4`: blast-radius minimization

This matters because an incident can look "resolved" while still being wrong in important ways. A model that closes quickly but misdiagnoses the cause should not score like a model that actually understands the failure. Likewise, a model that restores one service while degrading three more should be penalized.

SENTINEL also applies shaping penalties for harmful behavior, including blast-radius expansion and restarting healthy services.

## Failure Library

The scenario library covers operationally meaningful failure modes such as:

- memory leaks
- CPU spikes
- bad deployments
- connection pool exhaustion
- cache miss storms
- network partitions

Scenarios are distributed across easy, medium, and hard tiers. Harder incidents increase ambiguity through missing evidence, longer propagation chains, and red-herring signals.

## Training Setup

The repo includes a runnable training path in both script and notebook form:

- [`train.py`](train.py)
- [`retrain.py`](retrain.py)
- [`sentinel_colab_training.ipynb`](sentinel_colab_training.ipynb)
- [`TRAINING.md`](TRAINING.md)

The reported runs used `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` with LoRA adapters and a reward-driven RL loop. The training path is designed so judges can inspect the code and re-run it end to end from the repo instead of relying on external dashboards.

## Results

The strongest outcome in this submission is that the environment produces a learnable signal for diagnosis and remediation rather than only for generic tool use.

Headline metrics from the committed results:

- `holmes` reached `0.74` total reward on easy evaluation with `0.67` root-cause accuracy
- `forge` reached `0.82` total reward on easy evaluation and consistently reduced blast radius
- all five agents have committed reward and loss curves in the repo

### Training Curves

![All Agents Comparison](results/comparison_all_agents.png)

![Loss Comparison](results/comparison_loss.png)

### Per-Agent Curves

![Holmes Training](results/holmes_training_curves.png)
![Holmes Loss](results/holmes_loss_curve.png)

![Forge Training](results/forge_training_curves.png)
![Forge Loss](results/forge_loss_curve.png)

![Argus Training](results/argus_training_curves.png)
![Argus Loss](results/argus_loss_curve.png)

![Hermes Training](results/hermes_training_curves.png)
![Hermes Loss](results/hermes_loss_curve.png)

![Oracle Training](results/oracle_training_curves.png)
![Oracle Loss](results/oracle_loss_curve.png)

## What Makes SENTINEL Different

### 1. It is a real control problem

The model acts inside a stateful system. Observations change because of previous actions. Errors compound. Good remediation shrinks the future search space.

### 2. It separates diagnosis from remediation

Many benchmarks collapse the whole task into one success label. SENTINEL isolates the hard part: identifying the real fault before acting. That is much closer to real incident response.

### 3. It is submission-complete

The environment is packaged for OpenEnv-style validation, includes a parseable manifest, includes committed training plots, and includes both a notebook and script-based training path.

## Deliverables

- repository: [github.com/sayantikalaskar/sentinel](https://github.com/sayantikalaskar/sentinel)
- Space README entrypoint: [README.md](README.md)
- blog writeup: [Blog.MD](Blog.MD)
- training notebook: [sentinel_colab_training.ipynb](sentinel_colab_training.ipynb)
- training scripts: [train.py](train.py), [retrain.py](retrain.py)
- manifest: [openenv.yaml](openenv.yaml)
- result artifacts: [results/](results/)

## Final Note

SENTINEL is not trying to be another static benchmark with operational language on top. It is an attempt to turn cloud incident response into a trainable environment where success depends on causal reasoning, safe intervention, and long-horizon recovery.

That is the benchmark we wanted to train on, and it is the benchmark this repo now exposes cleanly for OpenEnv-style evaluation.
