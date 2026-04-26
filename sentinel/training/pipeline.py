"""GRPO training pipeline for SENTINEL.

This is the active training path:
  - action selection is driven by an LLMAgent during rollouts
  - GRPO uses SENTINEL rewards as the optimisation signal
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
    max_completion_length: int = 96
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

    # ── 3. Create optimizer for reward-weighted gradient steps ─────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-6, weight_decay=0.01)
    logger.info("Optimizer: AdamW over %d trainable params", len(trainable_params))

    # ── 4. LLMAgent for action generation during rollouts ─────────────────
    device = "cuda"
    FastLanguageModel.for_inference(model)
    llm_agent = LLMAgent(
        model=model,
        tokenizer=tokenizer,
        agent_role=agent,
        device=device,
    )
    # Attach optimizer to llm_agent so training loop can access it
    llm_agent.optimizer = optimizer
    logger.info("build_grpo_trainer: ready (model=%s, device=%s)", config.model_name, device)

    # Return (None, llm_agent) — trainer is no longer used
    return None, llm_agent


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
    """Run the GRPO training loop from *start_episode* to *config.max_steps*."""
    _ensure_llm_agent(llm_agent)
    all_metrics: list[EpisodeMetrics] = []
    logger.info(
        "Starting training loop: episodes=%d, mode=%s, start=%d",
        config.max_steps, "LLM", start_episode,
    )

    for episode in range(start_episode, config.max_steps):
        # Clear CUDA cache between episodes to prevent memory fragmentation
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
            err_msg = str(exc)
            if "CUDA out of memory" in err_msg:
                new_bs = max(1, config.batch_size // 2)
                logger.warning(
                    "CUDA OOM at episode %d \u2014 halving batch_size %d\u2192%d and retrying.",
                    episode, config.batch_size, new_bs,
                )
                config.batch_size = new_bs
                if trainer is not None and hasattr(trainer, "args"):
                    trainer.args.per_device_train_batch_size = new_bs
            elif "device-side assert" in err_msg or "cudaErrorAssert" in err_msg:
                import torch
                logger.warning(
                    "CUDA assert at episode %d \u2014 resetting CUDA state and retrying.",
                    episode,
                )
                torch.cuda.empty_cache()
                # Return degraded metrics instead of crashing
                return EpisodeMetrics(
                    episode=episode, r1=0.0, r2=0.0, r3=0.0, r4=0.0,
                    total_reward=0.0, mttr=0, loss=None,
                )
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
                logger.warning(
                    "Parser failure rate too high (%.2f) at episode %d step %d; ending episode early.",
                    failure_rate, episode, step_count,
                )
                break

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

    # ─ Reward-weighted gradient step (replaces GRPOTrainer.train()) ──────
    loss: float | None = None
    # Only do gradient step if we have enough valid (non-failed) samples
    valid_samples = [s for s in grpo_samples if not s.get("parse_failed")]
    episode_reward = breakdown.total  # Use episode-level reward for gradient weighting

    if hasattr(llm_agent, 'optimizer') and len(valid_samples) >= 1:
        try:
            import torch
            # Verify CUDA is still healthy before attempting training
            if torch.cuda.is_available():
                torch.cuda.current_device()  # will raise if CUDA state is bad

            model = llm_agent.model
            tokenizer = llm_agent.tokenizer
            optimizer = llm_agent.optimizer

            # Running baseline: exponential moving average of episode rewards
            if not hasattr(llm_agent, '_reward_baseline'):
                llm_agent._reward_baseline = episode_reward
            baseline = llm_agent._reward_baseline
            advantage = episode_reward - baseline
            # Update baseline with EMA (alpha=0.1)
            llm_agent._reward_baseline = 0.9 * baseline + 0.1 * episode_reward

            # Only do gradient step if advantage is meaningfully non-zero
            if abs(advantage) > 0.01:
                FastLanguageModel.for_training(model)
                optimizer.zero_grad()

                total_loss = 0.0
                n = min(len(valid_samples), config.batch_size)
                samples_batch = valid_samples[:n]

                # Clamp advantage to prevent extreme updates
                advantage = max(-2.0, min(2.0, advantage))

                for sample in samples_batch:
                    full_text = sample["prompt"] + sample["completion"]
                    enc = tokenizer(
                        full_text, return_tensors="pt",
                        truncation=True, max_length=config.max_seq_length,
                    ).to("cuda")

                    # Mask prompt tokens with -100 so loss is only on completion
                    prompt_enc = tokenizer(
                        sample["prompt"], return_tensors="pt",
                        truncation=True, max_length=config.max_seq_length,
                    )
                    prompt_len = prompt_enc["input_ids"].shape[1]
                    labels = enc["input_ids"].clone()
                    labels[:, :prompt_len] = -100

                    outputs = model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        labels=labels,
                    )

                    step_loss = outputs.loss * advantage / n
                    step_loss.backward()
                    total_loss += step_loss.item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                loss = total_loss

                FastLanguageModel.for_inference(model)

        except Exception as exc:
            logger.warning("Training step failed at episode %d: %s", episode, exc)
            # Clear CUDA state to prevent cascading failures
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                FastLanguageModel.for_inference(llm_agent.model)
            except Exception:
                pass

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
