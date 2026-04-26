#!/usr/bin/env python
"""
SENTINEL — Single-agent training worker.
Called by run_full_training.py in a subprocess for CUDA isolation.

Usage: python _train_worker.py <agent> <episodes> <tag>
"""
from __future__ import annotations

import json
import gc
import logging
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sentinel.worker")

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
BATCH_SIZE = 2
EVAL_EPISODES = 3


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <agent> <episodes> <tag>", file=sys.stderr)
        sys.exit(1)

    agent = sys.argv[1]
    episodes = int(sys.argv[2])
    tag = sys.argv[3]

    log_file = RESULTS_DIR / f"{agent}_{tag}_log.jsonl"
    ckpt_dir = PROJECT_ROOT / "checkpoints" / agent / tag

    # Clean previous artifacts
    if log_file.exists():
        log_file.unlink()
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    logger.info("=" * 60)
    logger.info("  Training: agent=%s | episodes=%d | tag=%s", agent, episodes, tag)
    logger.info("=" * 60)

    from sentinel.env import Sentinel_Env
    from sentinel.training.evaluate import print_eval_report, run_evaluation
    from sentinel.training.pipeline import (
        TrainingConfig,
        build_grpo_trainer,
        run_training_loop,
    )

    env = Sentinel_Env(config_path="env_spec.yaml")
    reward_fn = env.reward_function

    config = TrainingConfig(
        agent=agent,
        model_name=MODEL,
        load_in_4bit=True,
        batch_size=BATCH_SIZE,
        max_steps=episodes,
        lora_r=16,
        lora_alpha=32,
        checkpoint_dir=str(ckpt_dir),
        log_file=str(log_file),
    )

    t0 = time.perf_counter()
    trainer, llm_agent = build_grpo_trainer(agent=agent, env=env, config=config)

    all_metrics = run_training_loop(
        trainer=trainer,
        env=env,
        config=config,
        reward_fn=reward_fn,
        start_episode=0,
        llm_agent=llm_agent,
    )
    elapsed = time.perf_counter() - t0

    # Compute summary
    summary = {
        "agent": agent,
        "tag": tag,
        "episodes": len(all_metrics),
        "elapsed_s": round(elapsed, 1),
        "sec_per_ep": round(elapsed / max(len(all_metrics), 1), 1),
    }

    if all_metrics:
        last_n = all_metrics[-min(10, len(all_metrics)):]
        summary["avg_reward_last10"] = round(sum(m.total_reward for m in last_n) / len(last_n), 4)
        summary["avg_r1_last10"] = round(sum(m.r1 for m in last_n) / len(last_n), 4)
        summary["avg_mttr_last10"] = round(sum(m.mttr for m in last_n) / len(last_n), 1)
        summary["best_reward"] = round(max(m.total_reward for m in all_metrics), 4)
        summary["best_r1"] = round(max(m.r1 for m in all_metrics), 4)

    logger.info("Training done: %s", json.dumps(summary, indent=2))

    # Save summary BEFORE eval (eval may crash on CUDA cleanup)
    summary_file = RESULTS_DIR / f"{agent}_{tag}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved (pre-eval): %s", summary_file)

    # Run evaluation
    logger.info("Running post-training evaluation (%d eps/tier) ...", EVAL_EPISODES)
    try:
        eval_results = run_evaluation(
            env, reward_fn,
            llm_agent=llm_agent,
            episodes_per_tier=EVAL_EPISODES,
            seed=42,
        )
        print_eval_report(eval_results)
        summary["eval"] = {}
        for tier, data in eval_results.items():
            if hasattr(data, "r1_mean"):
                summary["eval"][tier] = {
                    "r1_mean": round(data.r1_mean, 4),
                    "total_mean": round(data.total_reward_mean, 4),
                    "mttr_mean": round(data.mttr_mean, 1),
                }
    except Exception as exc:
        logger.warning("Evaluation failed: %s", exc)

    # Update summary with eval results
    summary_file = RESULTS_DIR / f"{agent}_{tag}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved: %s", summary_file)

    # Print for parent process to capture
    print(f"__SUMMARY__{json.dumps(summary)}__END__")

    # Force-exit to avoid CUDA tensor cleanup crash (SIGABRT)
    # Python cleanup triggers destructor on poisoned CUDA tensors
    import os as _os
    _os.sync()  # flush file buffers
    sys.stdout.flush()
    sys.stderr.flush()
    _os._exit(0)


if __name__ == "__main__":
    main()
