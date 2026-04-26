#!/usr/bin/env python
"""SENTINEL LLM training entry point."""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sentinel.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate a SENTINEL LLM agent via GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent", choices=["holmes", "forge", "argus", "hermes", "oracle"], default="holmes",
        help="Which agent role to train.",
    )
    parser.add_argument(
        "--model",
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="HuggingFace / Unsloth model name or local path.",
    )
    parser.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
        default=True,
        help="Load the base model in 4-bit quantized mode.",
    )
    parser.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit loading and use full-precision/bf16 weights instead.",
    )
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes.")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device GRPO batch size.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--log-file", default="training_log.jsonl", help="Episode metrics JSONL.")
    parser.add_argument("--env-spec", default="env_spec.yaml", help="Path to env_spec.yaml.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes per tier in evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("Importing SENTINEL modules …")
    from sentinel.env import Sentinel_Env
    from sentinel.training.evaluate import print_eval_report, run_evaluation
    from sentinel.training.pipeline import (
        TrainingConfig,
        build_grpo_trainer,
        load_latest_checkpoint,
        run_training_loop,
    )

    logger.info("Initialising environment from %s …", args.env_spec)
    env = Sentinel_Env(config_path=args.env_spec)
    reward_fn = env.reward_function

    config = TrainingConfig(
        agent=args.agent,
        model_name=args.model,
        load_in_4bit=args.load_in_4bit,
        batch_size=args.batch_size,
        max_steps=args.episodes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        checkpoint_dir=str(Path(args.checkpoint_dir) / args.agent),
        log_file=args.log_file,
    )

    logger.info("=" * 54)
    logger.info("  SENTINEL Mode : LLM (GRPO)")
    logger.info("  Agent         : %s", args.agent)
    logger.info("  Model         : %s", args.model)
    logger.info("  4-bit         : %s", "on" if args.load_in_4bit else "off")
    logger.info("=" * 54)

    trainer, llm_agent = build_grpo_trainer(
        agent=args.agent,
        env=env,
        config=config,
    )

    start_episode = 0
    if args.resume:
        ckpt = load_latest_checkpoint(config.checkpoint_dir)
        if ckpt:
            start_episode = ckpt.get("episode", 0) + 1
            logger.info("Resuming from episode %d …", start_episode)
        else:
            logger.info("No checkpoint found in %s; starting from episode 0.", config.checkpoint_dir)

    if args.eval_only:
        logger.info("Eval-only mode: running %d episodes per tier …", args.eval_episodes)
        results = run_evaluation(
            env,
            reward_fn,
            llm_agent=llm_agent,
            episodes_per_tier=args.eval_episodes,
            seed=args.seed,
        )
        print_eval_report(results)
        return 0

    logger.info(
        "Starting training | agent=%s | mode=LLM (GRPO) | episodes=%d | start=%d",
        args.agent, args.episodes, start_episode,
    )
    t0 = time.perf_counter()

    all_metrics = run_training_loop(
        trainer=trainer,
        env=env,
        config=config,
        reward_fn=reward_fn,
        start_episode=start_episode,
        llm_agent=llm_agent,
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Training complete: %d episodes in %.1f s (%.2f s/ep)",
        len(all_metrics), elapsed, elapsed / max(len(all_metrics), 1),
    )

    if all_metrics:
        last_10 = all_metrics[-10:]
        avg_reward = sum(m.total_reward for m in last_10) / len(last_10)
        avg_mttr = sum(m.mttr for m in last_10) / len(last_10)
        logger.info(
            "Last 10 episodes | avg_reward=%.3f | avg_MTTR=%.1f",
            avg_reward, avg_mttr,
        )

    logger.info("Running post-training evaluation (%d eps/tier) …", args.eval_episodes)
    results = run_evaluation(
        env,
        reward_fn,
        llm_agent=llm_agent,
        episodes_per_tier=args.eval_episodes,
        seed=args.seed,
    )
    print_eval_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
