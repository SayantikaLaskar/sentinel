"""GRPO training pipeline for SENTINEL.

This module is intentionally LLM-first:
  - action selection is driven by an LLMAgent during rollouts
  - GRPO uses SENTINEL rewards as the optimisation signal
  - there is no math-policy fallback inside the active training path
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from sentinel.config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports — unsloth / trl may not be installed
# ---------------------------------------------------------------------------

try:
    from unsloth import FastLanguageModel  # type: ignore
    _UNSLOTH_AVAILABLE = True
except Exception as _unsloth_err:  # NotImplementedError when no GPU, ImportError when not installed
    FastLanguageModel = None  # type: ignore
    _UNSLOTH_AVAILABLE = False
    # Warn only once at import time so the caller knows why LLM mode is disabled
    import warnings as _w
    _w.warn(
        f"unsloth unavailable ({type(_unsloth_err).__name__}: {_unsloth_err}). "
        "GPU training is disabled until unsloth is installed on a CUDA machine.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from trl import GRPOTrainer, GRPOConfig  # type: ignore
    _TRL_AVAILABLE = True
except Exception as _trl_err:  # ImportError when not installed
    GRPOTrainer = None  # type: ignore
    GRPOConfig = None  # type: ignore
    _TRL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    agent: AgentRole
    model_name: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
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
    max_completion_length: int = 128
    parser_failure_abort_rate: float = 0.6
    parser_failure_warmup_steps: int = 8


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
    agent: AgentRole,
    env: Any,
    config: TrainingConfig,
) -> tuple[Any, Any]:
    """Build a GRPOTrainer + LLMAgent for the given SENTINEL agent role.

    Full LLM integration flow:
      1. Load ``unsloth/Meta-Llama-3-8B-Instruct`` (or config.model_name) in 4-bit
      2. Apply LoRA adapters (r=16, alpha=32) via Unsloth
      3. Wrap SENTINEL's Reward_Function as a GRPO reward signal
      4. Construct LLMAgent (obs→prompt→model→parse→action)
      5. Return (GRPOTrainer, LLMAgent)

    Returns:
      (trainer, llm_agent)
    """
    from sentinel.training.llm_agent import LLMAgent, make_grpo_reward_fn
    from sentinel.training.prompt_builder import build_messages

    if not _UNSLOTH_AVAILABLE or not _TRL_AVAILABLE:
        raise RuntimeError(
            "unsloth and trl are required for the LLM-only training path."
        )
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for the LLM-only training path."
        )

    # ── 1. Load base model in 4-bit quantization ──────────────────────────
    quant_mode = "4-bit" if config.load_in_4bit else "full precision"
    logger.info("Loading %s in %s ...", config.model_name, quant_mode)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=None,           # auto-detect bf16/fp16
    )

    # ── 2. Apply LoRA adapters ─────────────────────────────────────────────
    logger.info("Applying LoRA (r=%d, alpha=%d) ...", config.lora_r, config.lora_alpha)
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",  # unsloth's memory-efficient variant
        random_state=42,
    )

    # ── 3. GRPO reward function ────────────────────────────────────────────
    # GRPOTrainer expects: reward_fn(prompts, completions, **kwargs) -> list[float]
    grpo_reward_fn = make_grpo_reward_fn(env)

    # ── 4. Build a minimal prompt dataset for GRPOTrainer ─────────────────
    # TRL's GRPOTrainer needs a dataset of prompts to sample from.
    # We generate synthetic prompts from a fresh env reset.
    try:
        from datasets import Dataset  # type: ignore
        env_obs, _ = env.reset()
        sample_messages = build_messages(env_obs, agent_role=agent, step_number=0)
        sample_prompt = tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=True
        )
        # Create a small dataset; the real training data comes from env rollouts
        prompt_dataset = Dataset.from_dict({"prompt": [sample_prompt] * 8})
    except Exception as exc:
        logger.warning("Could not build prompt dataset: %s", exc)
        prompt_dataset = None

    # ── 5. GRPOTrainer config ─────────────────────────────────────────────
    grpo_cfg = GRPOConfig(
        output_dir=config.checkpoint_dir,
        per_device_train_batch_size=config.batch_size,
        max_steps=config.max_steps,
        num_generations=2,
        max_completion_length=config.max_completion_length,
        temperature=0.1,
        top_p=1.0,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        bf16=True,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[grpo_reward_fn],
        args=grpo_cfg,
        train_dataset=prompt_dataset,
    )

    # ── 6. LLMAgent for action generation during rollouts ─────────────────
    device = "cuda"
    llm_agent = LLMAgent(
        model=model,
        tokenizer=tokenizer,
        agent_role=agent,
        device=device,
    )
    logger.info("build_grpo_trainer: ready (model=%s, device=%s)", config.model_name, device)

    return trainer, llm_agent


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
    llm_agent: Any = None,
) -> list[EpisodeMetrics]:
    """Run the LLM-only training loop from *start_episode* to *config.max_steps*."""
    _ensure_llm_agent(llm_agent)
    all_metrics: list[EpisodeMetrics] = []
    logger.info(
        "Starting training loop: episodes=%d, mode=LLM, start=%d",
        config.max_steps, start_episode,
    )

    for episode in range(start_episode, config.max_steps):
        metrics = _run_single_episode(
            episode=episode,
            trainer=trainer,
            env=env,
            config=config,
            reward_fn=reward_fn,
            llm_agent=llm_agent,
        )
        all_metrics.append(metrics)
        log_episode_metrics(metrics, config.log_file)

        logger.info(
            "Episode %4d | R1=%.2f R2=%.2f R3=%.2f R4=%.2f | Total=%.3f | MTTR=%d | loss=%s",
            episode, metrics.r1, metrics.r2, metrics.r3, metrics.r4,
            metrics.total_reward, metrics.mttr,
            f"{metrics.loss:.4f}" if metrics.loss is not None else "n/a",
        )

        if episode % 10 == 0:
            state = {
                "episode": episode,
                "batch_size": config.batch_size,
                "mode": "LLM",
            }
            save_checkpoint(state, config.checkpoint_dir, episode)

    return all_metrics


def _run_single_episode(
    episode: int,
    trainer: Any,
    env: Any,
    config: TrainingConfig,
    reward_fn: Any,
    llm_agent: Any = None,
) -> EpisodeMetrics:
    """Run one episode, retrying on CUDA OOM by halving batch size."""
    while True:
        try:
            return _execute_episode(
                episode, trainer, env, config, reward_fn, llm_agent=llm_agent
            )
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                new_bs = max(1, config.batch_size // 2)
                logger.warning(
                    "CUDA OOM at episode %d \u2014 halving batch_size %d\u2192%d and retrying.",
                    episode, config.batch_size, new_bs,
                )
                config.batch_size = new_bs
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
    llm_agent: Any = None,
) -> EpisodeMetrics:
    """Execute a single episode and return its metrics.

    Generates actions via LLM inference (obs → prompt → model.generate → parse → action).
    When trainer is provided, runs a GRPOTrainer gradient step using the
    collected (prompt, completion, reward) triples.
    """
    from sentinel.training.prompt_builder import build_prompt
    from sentinel.models import Action, RewardBreakdown, Trajectory, TrajectoryStep

    _ensure_llm_agent(llm_agent)
    obs, info = env.reset()
    steps: list[TrajectoryStep] = []
    terminated = False
    truncated = False
    step_count = 0

    # Collect (prompt, completion, step_reward) for GRPO update
    grpo_samples: list[dict] = []

    # Reset LLM agent state for new episode
    llm_agent.reset()

    while not (terminated or truncated):
        # ─ Build text prompt for this observation ────────────────────────────
        text_prompt = build_prompt(
            obs,
            agent_role=getattr(llm_agent, "agent_role", config.agent),
            step_number=step_count,
        )

        # ─ Select action ──────────────────────────────────────────────────────
        action_dict = llm_agent.act(obs, step=step_count)
        llm_completion = action_dict.pop("_llm_completion", None)
        parse_failed = bool(action_dict.pop("_parse_failed", False))

        obs_next, reward, terminated, truncated, step_info = env.step(action_dict)
        step_info["parse_failed"] = parse_failed

        # ─ Record GRPO sample ─────────────────────────────────────────────
        if llm_completion is not None:
            grpo_samples.append({
                "prompt":     text_prompt,
                "completion": llm_completion,
                "reward":     float(reward),
                "parse_failed": parse_failed,
            })

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

        if step_count >= config.parser_failure_warmup_steps:
            parse_failures = sum(1 for sample in grpo_samples if sample.get("parse_failed"))
            failure_rate = parse_failures / max(len(grpo_samples), 1)
            if failure_rate >= config.parser_failure_abort_rate:
                raise RuntimeError(
                    f"Parser failure rate too high ({failure_rate:.2f}) at episode {episode}; aborting early to avoid wasting GPU."
                )

    # ─ Build trajectory ─────────────────────────────────────────────────
    episode_id = info.get("incident_id", str(uuid.uuid4()))
    incident_template_id = info.get("incident_id", "unknown")
    mttr = step_count

    incident_state = env._incident_state
    world_state = env.world_state

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

    # ─ GRPO gradient step (if trainer + samples available) ────────────────
    loss: float | None = None
    if trainer is not None and grpo_samples:
        try:
            from datasets import Dataset  # type: ignore
            grpo_ds = Dataset.from_list([{
                "prompt":     s["prompt"],
                "completion": s["completion"],
            } for s in grpo_samples])
            trainer.train_dataset = grpo_ds
            result = trainer.train()
            if hasattr(result, "training_loss"):
                loss = float(result.training_loss)
        except Exception as exc:
            logger.warning("GRPO trainer step failed at episode %d: %s", episode, exc)

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


def _ensure_llm_agent(llm_agent: Any) -> None:
    if llm_agent is None:
        raise RuntimeError(
            "run_training_loop requires an llm_agent in the LLM-only training path."
        )

AgentRole = Literal["holmes", "forge", "argus", "hermes", "oracle"]
