#!/usr/bin/env python3
"""SENTINEL — Retrain both agents (Holmes + Forge) with fixed env/reward.
Runs _train_worker.py for each agent sequentially, then generates plots.
"""
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrain")

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
EPISODES = 100
AGENTS = ["holmes", "forge", "argus", "hermes", "oracle"]


def run_agent(agent: str) -> dict | None:
    """Run _train_worker.py for one agent in a subprocess."""
    logger.info("=" * 60)
    logger.info("  TRAINING: %s | %d episodes", agent.upper(), EPISODES)
    logger.info("=" * 60)

    log_file = RESULTS_DIR / f"{agent}_retrain.log"
    t0 = time.perf_counter()

    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("PYTHONPATH", None)

    with open(log_file, "w") as fh:
        result = subprocess.run(
            [VENV_PYTHON, "_train_worker.py", agent, str(EPISODES), "full"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=7200,  # 2hr safety
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "%s finished: exit_code=%d elapsed=%.0fs (%.1f min)",
        agent.upper(), result.returncode, elapsed, elapsed / 60,
    )

    # Check summary file
    summary_file = RESULTS_DIR / f"{agent}_full_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        logger.info("%s summary: %s", agent.upper(), json.dumps(summary, indent=2))
        return summary
    else:
        logger.error("No summary file for %s!", agent)
        return None


def main():
    logger.info("SENTINEL RETRAIN — Starting both agents sequentially")
    logger.info("Fixes applied: FormHypothesis terminates Holmes, blast_resolved terminates Forge")
    logger.info("Fixes applied: Advantage-weighted gradient steps, improved prompts")

    summaries = {}
    for agent in AGENTS:
        try:
            summary = run_agent(agent)
            summaries[agent] = summary
        except subprocess.TimeoutExpired:
            logger.error("%s TIMED OUT after 2 hours", agent.upper())
            summaries[agent] = None
        except Exception as exc:
            logger.error("%s FAILED: %s", agent.upper(), exc)
            summaries[agent] = None

    # Generate plots
    logger.info("Generating training curves...")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [VENV_PYTHON, "generate_curves.py"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    # Save consolidated results
    results = {
        "project": "SENTINEL",
        "hackathon": "Meta PyTorch OpenEnv Hackathon 2026",
        "model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "gpu": "NVIDIA L40S (48GB)",
        "episodes_per_agent": EPISODES,
        "fixes_applied": [
            "FormHypothesis terminates Holmes episodes (variable MTTR)",
            "Blast radius=0 terminates Forge episodes (variable MTTR)",
            "Advantage-weighted REINFORCE gradient steps (proper credit assignment)",
            "Explicit prompts guide Holmes to FormHypothesis after investigation",
            "R2 MTTR penalized when R1=0 (no root cause identified)",
            "Forge gets partial R1 credit from remediation target inference",
        ],
        "agents": summaries,
    }
    out_file = RESULTS_DIR / "retrain_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Consolidated results: %s", out_file)

    logger.info("=" * 60)
    logger.info("  ALL DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
