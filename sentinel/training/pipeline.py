"""GRPO training pipeline for SENTINEL.

Provides:
- TrainingConfig: dataclass for all training hyperparameters
- EpisodeMetrics: dataclass for per-episode logging
- build_grpo_trainer(): load model + LoRA + GRPOTrainer (graceful fallback)
- log_episode_metrics(): append JSON line to log file
- save_checkpoint() / load_latest_checkpoint(): persistent episode counter
- run_training_loop(): full training loop with OOM handling and checkpointing

Action selection priority (highest to lowest):
  1. GRPOTrainer.generate()       — full fine-tuned model (GPU required)
  2. UCB1 + Bayesian RCA          — observation-driven math (no GPU, no API)
     UCB1: Auer, Cesa-Bianchi & Fischer (2002). Machine Learning, 47, 235-256.
     Bayes: Noisy-OR belief propagation (Pearl 1988 / MicroRank WWW 2021).
"""
from __future__ import annotations

import json
import logging
import os
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

from sentinel.config import load_config
from sentinel.math_engine import get_bayesian_rca, get_ucb1_bandit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports — unsloth / trl may not be installed
# ---------------------------------------------------------------------------

try:
    from unsloth import FastLanguageModel  # type: ignore
    _UNSLOTH_AVAILABLE = True
except ImportError:
    FastLanguageModel = None  # type: ignore
    _UNSLOTH_AVAILABLE = False

try:
    from trl import GRPOTrainer, GRPOConfig  # type: ignore
    _TRL_AVAILABLE = True
except ImportError:
    GRPOTrainer = None  # type: ignore
    GRPOConfig = None  # type: ignore
    _TRL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    agent: Literal["holmes", "forge"]
    model_name: str = "unsloth/Meta-Llama-3-8B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    batch_size: int = 4
    max_steps: int = 1000
    checkpoint_dir: str = "checkpoints"
    log_file: str = "training_log.jsonl"
    sla_breach_threshold: int = 50


@dataclass
class EpisodeMetrics:
    episode: int
    r1: float
    r2: float
    r3: float
    r4: float
    total_reward: float
    mttr: int
    loss: float | None = None


# ---------------------------------------------------------------------------
# Trainer construction
# ---------------------------------------------------------------------------

def build_grpo_trainer(
    agent: Literal["holmes", "forge"],
    env: Any,
    config: TrainingConfig,
) -> Any:
    """Build and return a GRPOTrainer for the given agent.

    Loads ``unsloth/Meta-Llama-3-8B-Instruct`` in 4-bit, applies LoRA
    (r=16, alpha=32), and wraps it in a GRPOTrainer that uses
    ``Reward_Function.compute_episode_reward`` as its sole reward signal.

    Returns ``None`` when unsloth or trl are unavailable (graceful degradation).
    """
    if not _UNSLOTH_AVAILABLE or not _TRL_AVAILABLE:
        warnings.warn(
            "unsloth and/or trl are not installed — build_grpo_trainer() returning None. "
            "Training will run in simulation mode.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    # Load base model in 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=True,
    )

    # Reward wrapper: delegate to env.reward_function.compute_episode_reward
    def _reward_fn(trajectory, world_state=None, incident_state=None):
        rf = env.reward_function
        ws = world_state or env.world_state
        inc = incident_state or env._incident_state
        breakdown = rf.compute_episode_reward(trajectory, ws, inc)
        return breakdown.total

    grpo_cfg = GRPOConfig(
        per_device_train_batch_size=config.batch_size,
        max_steps=config.max_steps,
        output_dir=config.checkpoint_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[_reward_fn],
        args=grpo_cfg,
    )
    return trainer


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_episode_metrics(metrics: EpisodeMetrics, log_file: str) -> None:
    """Append a JSON line with all EpisodeMetrics fields to *log_file*."""
    record = {
        "episode": metrics.episode,
        "r1": metrics.r1,
        "r2": metrics.r2,
        "r3": metrics.r3,
        "r4": metrics.r4,
        "total_reward": metrics.total_reward,
        "mttr": metrics.mttr,
        "loss": metrics.loss,
    }
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, checkpoint_dir: str, episode: int) -> None:
    """Save *state* as JSON to ``{checkpoint_dir}/checkpoint_{episode:06d}.json``.

    Creates *checkpoint_dir* if it does not exist.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_{episode:06d}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh)


def load_latest_checkpoint(checkpoint_dir: str) -> dict | None:
    """Return the most recent valid checkpoint dict, or ``None``.

    If the latest checkpoint file is corrupted (JSON parse error), falls back
    to the previous one and logs a warning.  Returns ``None`` when no
    checkpoints exist.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    files = sorted(
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".json")
    )
    if not files:
        return None

    # Try from newest to oldest
    for filename in reversed(files):
        path = os.path.join(checkpoint_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Checkpoint %s is corrupted (%s); falling back to previous checkpoint.",
                filename,
                exc,
            )

    return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def get_heuristic_action(obs: dict, agent_role: str = "holmes") -> dict:
    """Return an observation-driven action using the HOLMES or FORGE heuristic agent.

    This replaces the old static placeholder — actions are now derived from
    what the observation actually shows (blast radius, alerts, hypotheses).
    """
    import json as _json

    # Parse incident context from observation
    incident_ctx_raw = obs.get("incident_context", "{}")
    if isinstance(incident_ctx_raw, str):
        try:
            incident_ctx = _json.loads(incident_ctx_raw)
        except _json.JSONDecodeError:
            incident_ctx = {}
    else:
        incident_ctx = incident_ctx_raw

    blast_radius: list[str] = incident_ctx.get("current_blast_radius", [])
    hypotheses: list[dict] = incident_ctx.get("active_hypotheses", [])

    # Parse active alerts
    alerts_raw = obs.get("active_alerts", "[]")
    if isinstance(alerts_raw, str):
        try:
            alerts = _json.loads(alerts_raw)
        except _json.JSONDecodeError:
            alerts = []
    else:
        alerts = alerts_raw

    alert_service_counts: dict[str, int] = {}
    for a in alerts:
        svc = a.get("service", "") if isinstance(a, dict) else getattr(a, "service", "")
        if svc:
            alert_service_counts[svc] = alert_service_counts.get(svc, 0) + 1

    # Pick most-alerted service or first blast-radius service
    if alert_service_counts:
        top_service = max(alert_service_counts, key=alert_service_counts.__getitem__)
    elif blast_radius:
        top_service = blast_radius[0]
    else:
        top_service = "api-gateway"  # safe default that always exists in NexaStack

    if agent_role == "forge":
        # FORGE: remediate the highest-confidence hypothesis or most-alerted service
        if hypotheses:
            best = max(
                hypotheses,
                key=lambda h: h.get("confidence", 0.0) if isinstance(h, dict)
                else getattr(h, "confidence", 0.0),
            )
            target = best.get("service", top_service) if isinstance(best, dict) else getattr(best, "service", top_service)
        else:
            target = top_service
        return {
            "agent": "forge",
            "category": "remediation",
            "name": "RestartService",
            "params": {"service": target},
        }

    # Default: HOLMES investigates the most-alerted service
    return {
        "agent": "holmes",
        "category": "investigative",
        "name": "QueryLogs",
        "params": {"service": top_service, "time_range": [0, 300]},
    }


# Keep get_placeholder_action as an alias for backward compatibility (demo/app.py)
def get_placeholder_action(config_path: str = "env_spec.yaml") -> dict[str, Any]:
    """Backward-compatible alias — returns config-driven action for initial demo seeding."""
    try:
        cfg = load_config(config_path)
        return dict(cfg.training.placeholder_action)
    except Exception:
        return {
            "agent": "holmes",
            "category": "investigative",
            "name": "QueryMetrics",
            "params": {"service": "api-gateway", "metric_name": "error_rate", "time_range": [0, 300]},
        }


def run_training_loop(
    trainer: Any,
    env: Any,
    config: TrainingConfig,
    reward_fn: Any,
    start_episode: int = 0,
) -> list[EpisodeMetrics]:
    """Run the GRPO training loop from *start_episode* to *config.max_steps*.

    For each episode:
    1. Reset the environment.
    2. Step until terminated or truncated, collecting a Trajectory.
    3. Compute episode reward via ``reward_fn.compute_episode_reward``.
    4. Log metrics and save checkpoint every 10 episodes.

    CUDA OOM is handled by halving ``config.batch_size`` and retrying.
    Works even when *trainer* is ``None`` (no GPU / unsloth).
    """
    from sentinel.models import Action, RewardBreakdown, Trajectory, TrajectoryStep

    all_metrics: list[EpisodeMetrics] = []

    for episode in range(start_episode, config.max_steps):
        metrics = _run_single_episode(
            episode=episode,
            trainer=trainer,
            env=env,
            config=config,
            reward_fn=reward_fn,
        )
        all_metrics.append(metrics)
        log_episode_metrics(metrics, config.log_file)

        if episode % 10 == 0:
            state = {
                "episode": episode,
                "batch_size": config.batch_size,
            }
            save_checkpoint(state, config.checkpoint_dir, episode)

    return all_metrics


def _run_single_episode(
    episode: int,
    trainer: Any,
    env: Any,
    config: TrainingConfig,
    reward_fn: Any,
) -> EpisodeMetrics:
    """Run one episode, retrying on CUDA OOM by halving batch size."""
    while True:
        try:
            return _execute_episode(episode, trainer, env, config, reward_fn)
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                new_bs = max(1, config.batch_size // 2)
                logger.warning(
                    "CUDA OOM at episode %d — halving batch_size from %d to %d and retrying.",
                    episode,
                    config.batch_size,
                    new_bs,
                )
                config.batch_size = new_bs
                # Update trainer batch size if possible
                if trainer is not None and hasattr(trainer, "args"):
                    trainer.args.per_device_train_batch_size = new_bs
            else:
                raise


def _execute_episode(
    episode: int,
    trainer: Any,
    env: Any,
    config: TrainingConfig,
    reward_fn: Any,
) -> EpisodeMetrics:
    """Execute a single episode and return its metrics."""
    from sentinel.models import Action, RewardBreakdown, Trajectory, TrajectoryStep

    obs, info = env.reset()
    steps: list[TrajectoryStep] = []
    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated):
        # Use trainer to get action when available; otherwise use placeholder
        action_dict = _get_action(trainer, obs, config)

        pre_incident_state = env._incident_state

        obs_next, reward, terminated, truncated, step_info = env.step(action_dict)

        try:
            parsed_action = Action(
                agent=action_dict["agent"],
                category=action_dict["category"],
                name=action_dict["name"],
                params=action_dict.get("params", {}),
            )
        except Exception:
            parsed_action = Action(
                agent="holmes",
                category="investigative",
                name="QueryLogs",
                params={},
            )

        steps.append(
            TrajectoryStep(
                observation=obs,
                action=parsed_action,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=step_info,
            )
        )
        obs = obs_next
        step_count += 1

    # Build trajectory
    episode_id = info.get("incident_id", str(uuid.uuid4()))
    incident_template_id = info.get("incident_id", "unknown")
    mttr = step_count

    # Compute episode reward
    incident_state = env._incident_state
    world_state = env.world_state

    # Ensure steps is non-empty for reward computation
    if not steps:
        breakdown = RewardBreakdown(r1=0.0, r2=0.0, r3=0.0, r4=0.0, penalties=0.0, total=0.0)
    else:
        trajectory = Trajectory(
            episode_id=str(episode_id),
            incident_template_id=str(incident_template_id),
            steps=steps,
            final_reward=RewardBreakdown(r1=0.0, r2=0.0, r3=0.0, r4=0.0, penalties=0.0, total=0.0),
            mttr=mttr,
        )
        if incident_state is not None:
            breakdown = reward_fn.compute_episode_reward(trajectory, world_state, incident_state)
        else:
            breakdown = RewardBreakdown(r1=0.0, r2=0.0, r3=0.0, r4=0.0, penalties=0.0, total=0.0)

    # Optionally run a trainer step
    loss: float | None = None
    if trainer is not None and hasattr(trainer, "train"):
        try:
            result = trainer.train()
            if hasattr(result, "training_loss"):
                loss = float(result.training_loss)
        except Exception as exc:
            logger.warning("Trainer step failed at episode %d: %s", episode, exc)

    return EpisodeMetrics(
        episode=episode,
        r1=breakdown.r1,
        r2=breakdown.r2,
        r3=breakdown.r3,
        r4=breakdown.r4,
        total_reward=breakdown.total,
        mttr=mttr,
        loss=loss,
    )


def _get_action(trainer: Any, obs: dict, config: TrainingConfig) -> dict:
    """Return an action dict using the best available method.

    Priority:
      1. GRPOTrainer.generate()  — full fine-tuned model
      2. UCB1 bandit + Bayesian RCA (math-only, no API needed)
    """
    # 1. Fine-tuned GRPO model
    if trainer is not None and hasattr(trainer, "generate"):
        try:
            return trainer.generate(obs)
        except Exception as exc:
            logger.debug("trainer.generate failed: %s", exc)

    # 2. UCB1 bandit selects action arm; Bayesian RCA fills params
    return _ucb1_math_action(obs, config)


def _ucb1_math_action(obs: dict, config: TrainingConfig) -> dict:
    """Select and fill an action using UCB1 bandit + Bayesian RCA.

    UCB1 (Auer et al. 2002) selects which action type to try next.
    Bayesian Noisy-OR (Pearl 1988) identifies the most likely root-cause
    service and fills action parameters accordingly.
    The chosen arm is updated with the step reward after env.step().
    """
    bandit = get_ucb1_bandit()
    rca = get_bayesian_rca()

    # UCB1: select arm
    arm_idx = bandit.select()
    action = bandit.get_action_template(arm_idx)

    # Bayesian RCA: identify top-k candidate services
    top_candidates = rca.top_k(obs, k=3)
    top_service = top_candidates[0][0] if top_candidates else "api-gateway"
    top_ft = _infer_failure_type(obs, top_service)

    # Fill params based on action name
    name = action.get("name", "QueryLogs")
    params = _fill_params(name, top_service, top_ft, obs)
    action["params"] = params

    # Attach arm_idx so the training loop can update the bandit after the step
    action["_ucb1_arm_idx"] = arm_idx
    return action


def _infer_failure_type(obs: dict, service: str) -> str:
    """Heuristically infer likely failure type from metric anomaly pattern."""
    metrics_raw = obs.get("metrics_snapshot", "{}")
    if isinstance(metrics_raw, str):
        try:
            snap = json.loads(metrics_raw)
        except json.JSONDecodeError:
            snap = {}
    else:
        snap = metrics_raw or {}

    m = snap.get(service)
    if not m:
        return "cpu_spike"
    if isinstance(m, dict):
        cpu = m.get("cpu", 0)
        mem = m.get("memory", 0)
        err = m.get("error_rate", 0)
        lat = m.get("latency_ms", 0)
        sat = m.get("saturation", 0)
    else:
        cpu = getattr(m, "cpu", 0)
        mem = getattr(m, "memory", 0)
        err = getattr(m, "error_rate", 0)
        lat = getattr(m, "latency_ms", 0)
        sat = getattr(m, "saturation", 0)

    # Simple threshold-based classification
    if cpu > 0.85:
        return "cpu_spike"
    if mem > 0.85:
        return "memory_leak"
    if err > 0.1 and lat > 300:
        return "bad_deployment"
    if sat > 0.9:
        return "connection_pool_exhaustion"
    if err > 0.05:
        return "cache_miss_storm"
    return "network_partition"


def _fill_params(name: str, service: str, failure_type: str, obs: dict) -> dict:
    """Fill action parameters from Bayesian RCA output."""
    if name == "QueryLogs":
        return {"service": service, "time_range": [0, 300]}
    if name == "QueryMetrics":
        _metric_map = {
            "cpu_spike": "cpu",
            "memory_leak": "memory",
            "bad_deployment": "error_rate",
            "connection_pool_exhaustion": "saturation",
            "cache_miss_storm": "error_rate",
            "network_partition": "latency_ms",
        }
        return {"service": service, "metric_name": _metric_map.get(failure_type, "error_rate"), "time_range": [0, 300]}
    if name == "QueryTrace":
        return {"trace_id": f"{service}-trace-001"}
    if name == "FormHypothesis":
        return {"service": service, "failure_type": failure_type, "confidence": 0.75}
    if name == "RestartService":
        return {"service": service}
    if name == "ScaleService":
        return {"service": service, "replicas": 3}
    if name == "RollbackDeployment":
        return {"service": service, "version": "previous"}
    if name == "DrainTraffic":
        return {"service": service}
    if name == "ModifyRateLimit":
        return {"service": service, "limit_rps": 100}
    if name == "CloseIncident":
        return {"resolution_summary": f"Root cause identified: {service} ({failure_type})"}
    if name == "EscalateToHuman":
        return {"reason": f"High uncertainty about root cause in {service}"}
    if name == "GenerateNewScenario":
        return {"difficulty": "medium", "target_gap": "investigative"}
    return {}

