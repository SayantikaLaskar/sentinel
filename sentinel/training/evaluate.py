"""Evaluation pipeline for SENTINEL.

Runs held-out episodes per difficulty tier using the active LLMAgent and
reports mean ± std for R1, R2, R3, R4, total reward, and MTTR.
"""
from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class EvalResult:
    difficulty: str
    n_episodes: int
    r1_mean: float
    r1_std: float
    r2_mean: float
    r2_std: float
    r3_mean: float
    r3_std: float
    r4_mean: float
    r4_std: float
    total_reward_mean: float
    total_reward_std: float
    mttr_mean: float
    mttr_std: float


def _run_single_eval_episode(
    env: Any,
    reward_fn: Any,
    difficulty: str,
    episode_seed: int,
    llm_agent: Any,
) -> tuple[float, float, float, float, float, float]:
    """Run one evaluation episode and return (r1, r2, r3, r4, total, mttr)."""
    from sentinel.models import Action, RewardBreakdown, Trajectory, TrajectoryStep

    llm_agent.reset()
    obs, info = env.reset(seed=episode_seed)
    steps: list[TrajectoryStep] = []
    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated):
        action_dict = dict(llm_agent.act(obs, step=step_count))
        action_dict.pop("_llm_completion", None)
        action_dict.pop("_parse_failed", None)
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

    incident_state = env._incident_state
    world_state = env.world_state

    if not steps or incident_state is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, float(step_count)

    episode_id = info.get("incident_id", str(uuid.uuid4()))
    trajectory = Trajectory(
        episode_id=str(episode_id),
        incident_template_id=str(episode_id),
        steps=steps,
        final_reward=RewardBreakdown(r1=0.0, r2=0.0, r3=0.0, r4=0.0, penalties=0.0, total=0.0),
        mttr=step_count,
    )

    breakdown = reward_fn.compute_episode_reward(trajectory, world_state, incident_state)
    return (
        breakdown.r1,
        breakdown.r2,
        breakdown.r3,
        breakdown.r4,
        breakdown.total,
        float(step_count),
    )


def run_evaluation(
    env: Any,
    reward_fn: Any,
    llm_agent: Any,
    episodes_per_tier: int = 10,
    seed: int = 0,
) -> dict[str, EvalResult]:
    """Run evaluation across all three difficulty tiers."""
    tiers = ["easy", "medium", "hard"]
    original_distribution = dict(env._difficulty_distribution)
    results: dict[str, EvalResult] = {}

    for tier_index, difficulty in enumerate(tiers):
        env._difficulty_distribution = {t: (1.0 if t == difficulty else 0.0) for t in tiers}

        r1s, r2s, r3s, r4s, totals, mttrs = [], [], [], [], [], []

        for episode_index in range(episodes_per_tier):
            episode_seed = seed + tier_index * episodes_per_tier + episode_index
            r1, r2, r3, r4, total, mttr = _run_single_eval_episode(
                env, reward_fn, difficulty, episode_seed, llm_agent
            )
            r1s.append(r1)
            r2s.append(r2)
            r3s.append(r3)
            r4s.append(r4)
            totals.append(total)
            mttrs.append(mttr)

        env._difficulty_distribution = dict(original_distribution)

        def _std(values: list[float]) -> float:
            return statistics.stdev(values) if len(values) > 1 else 0.0

        results[difficulty] = EvalResult(
            difficulty=difficulty,
            n_episodes=episodes_per_tier,
            r1_mean=statistics.mean(r1s),
            r1_std=_std(r1s),
            r2_mean=statistics.mean(r2s),
            r2_std=_std(r2s),
            r3_mean=statistics.mean(r3s),
            r3_std=_std(r3s),
            r4_mean=statistics.mean(r4s),
            r4_std=_std(r4s),
            total_reward_mean=statistics.mean(totals),
            total_reward_std=_std(totals),
            mttr_mean=statistics.mean(mttrs),
            mttr_std=_std(mttrs),
        )

    return results


def print_eval_report(results: dict[str, EvalResult]) -> None:
    """Print a formatted evaluation report table."""
    header = (
        f"{'Difficulty':<10} | {'R1':<12} | {'R2':<12} | {'R3':<12} | "
        f"{'R4':<12} | {'Total':<12} | {'MTTR':<12}"
    )
    separator = "-" * len(header)

    print("=== SENTINEL Evaluation Report ===")
    print(header)
    print(separator)

    for difficulty in ["easy", "medium", "hard"]:
        if difficulty not in results:
            continue
        r = results[difficulty]

        def _fmt(mean: float, std: float) -> str:
            return f"{mean:.2f} ± {std:.2f}"

        row = (
            f"{r.difficulty:<10} | "
            f"{_fmt(r.r1_mean, r.r1_std):<12} | "
            f"{_fmt(r.r2_mean, r.r2_std):<12} | "
            f"{_fmt(r.r3_mean, r.r3_std):<12} | "
            f"{_fmt(r.r4_mean, r.r4_std):<12} | "
            f"{_fmt(r.total_reward_mean, r.total_reward_std):<12} | "
            f"{r.mttr_mean:.1f} ± {r.mttr_std:.1f}"
        )
        print(row)
