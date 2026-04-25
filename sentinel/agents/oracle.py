"""ORACLE — Self-Improvement Agent for SENTINEL.

Analyzes completed incident trajectories to identify capability gaps,
stores trajectories in ChromaDB (falls back to in-memory),
generates new IncidentTemplates using ALP Curriculum Learning,
and retires below-median templates when library exceeds 50 entries.

Scenario generation is driven by Absolute Learning Progress (ALP):
  Portelas et al. (2020). "Teacher algorithms for curriculum learning
  of Deep RL in continuously parameterized environments." CoRL 2020.
  ALP_t(c) = |R_t(c) - R_{t-Δt}(c)|

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
"""
from __future__ import annotations

import logging
import random
import statistics
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING

from sentinel.agents.base import BaseAgent
from sentinel.math_engine import get_alp_curriculum, ALPCurriculum
from sentinel.models import Action, FailureType, IncidentTemplate, Trajectory
from sentinel.world_state import ALL_SERVICES

if TYPE_CHECKING:
    from sentinel.incident_generator import Incident_Generator

logger = logging.getLogger(__name__)

# Difficulty escalation ladder (Req 12.4)
_DIFFICULTY_ESCALATION: dict[str, str] = {
    "easy": "medium",
    "medium": "hard",
    "hard": "hard",  # capped at hard
}

# Try to import ChromaDB; fall back to in-memory on ImportError or connection failure
_CHROMADB_AVAILABLE = False
try:
    import chromadb  # type: ignore
    _CHROMADB_AVAILABLE = True
except ImportError:
    pass


class ORACLE(BaseAgent):
    """Self-Improvement agent implementing the Hyperagent principle (arXiv:2603.19461).

    Tracks ORACLE-generated templates and retires below-median utility ones
    when the count exceeds 50.
    """

    def __init__(
        self,
        incident_generator: "Incident_Generator | None" = None,
        chromadb_host: str = "localhost",
        chromadb_port: int = 8001,
    ) -> None:
        self.incident_generator = incident_generator

        # Template tracking (Req 12.7)
        self.oracle_template_count: int = 0
        self.oracle_template_utility: dict[str, float] = {}  # template_id -> utility

        # Trajectory storage
        self._trajectories: list[Trajectory] = []

        # ChromaDB setup (Req 12.2)
        self._chroma_client = None
        self._chroma_collection = None
        self._use_chromadb = False

        if _CHROMADB_AVAILABLE:
            try:
                self._chroma_client = chromadb.HttpClient(
                    host=chromadb_host, port=chromadb_port
                )
                self._chroma_collection = self._chroma_client.get_or_create_collection(
                    "sentinel_trajectories"
                )
                self._use_chromadb = True
            except Exception:
                # Fall back to in-memory (Req 12.2 error handling)
                self._use_chromadb = False

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: dict) -> Action:
        """Emit a GenerateNewScenario action based on identified capability gap."""
        gap = observation.get("capability_gap", "investigative")
        difficulty = observation.get("current_difficulty", "easy")
        next_difficulty = _DIFFICULTY_ESCALATION.get(difficulty, "hard")

        return Action(
            agent="oracle",
            category="meta",
            name="GenerateNewScenario",
            params={
                "difficulty": next_difficulty,
                "target_gap": gap,
            },
        )

    def reset(self) -> None:
        """Reset ORACLE episode state (does not clear template library)."""
        pass  # ORACLE persists state across episodes by design

    # ------------------------------------------------------------------
    # Trajectory analysis (Req 12.1)
    # ------------------------------------------------------------------

    def analyze_trajectory(self, trajectory: Trajectory) -> str:
        """Identify the action category with the worst performance.

        Returns the name of the worst-performing action category.
        """
        # Accumulate reward per action category
        category_rewards: dict[str, list[float]] = defaultdict(list)
        for step in trajectory.steps:
            cat = step.action.category
            category_rewards[cat].append(step.reward)

        if not category_rewards:
            return "investigative"

        # Worst = lowest mean reward
        worst_category = min(
            category_rewards,
            key=lambda c: statistics.mean(category_rewards[c]),
        )
        return worst_category

    # ------------------------------------------------------------------
    # Trajectory storage (Req 12.2)
    # ------------------------------------------------------------------

    def store_trajectory(self, trajectory: Trajectory) -> None:
        """Store a completed trajectory in ChromaDB or in-memory fallback."""
        self._trajectories.append(trajectory)

        if self._use_chromadb and self._chroma_collection is not None:
            try:
                self._chroma_collection.add(
                    documents=[trajectory.to_json()],
                    ids=[trajectory.episode_id],
                    metadatas=[{
                        "incident_template_id": trajectory.incident_template_id,
                        "mttr": trajectory.mttr,
                        "total_reward": trajectory.final_reward.total,
                    }],
                )
            except Exception:
                # Silently fall back to in-memory on connection failure
                pass

    # ------------------------------------------------------------------
    # Scenario generation (Req 12.3, 12.4, 12.5)
    # ------------------------------------------------------------------

    def generate_scenario(
        self,
        trajectory: Trajectory,
        source_template: IncidentTemplate | None = None,
    ) -> IncidentTemplate:
        """Generate a new IncidentTemplate using ALP Curriculum Learning.

        Algorithm (Portelas et al. 2020):
          1. Record this trajectory's total reward in the ALP curriculum.
          2. Call curriculum.next_task() to get (difficulty, failure_type)
             with highest |ALP| = |mean(recent) - mean(earlier)| rewards.
          3. Build a template targeting that (difficulty, failure_type) cell,
             escalating from the source template difficulty.

        Assigns difficulty one level above the source incident (capped at Hard)
        OR uses ALP-selected difficulty — whichever is harder.
        """
        gap = self.analyze_trajectory(trajectory)

        # Determine difficulty escalation (Req 12.4)
        source_difficulty = source_template.difficulty if source_template else "easy"
        escalated = _DIFFICULTY_ESCALATION.get(source_difficulty, "hard")

        # ALP-based task selection
        curriculum = get_alp_curriculum()

        # Record current episode performance
        if trajectory.steps:
            curriculum.record(
                difficulty=source_difficulty,
                failure_type=source_template.failure_type.value if source_template else "cpu_spike",
                total_reward=trajectory.final_reward.total,
            )

        # Get ALP-maximising next task
        alp_difficulty, alp_failure_type_str = curriculum.next_task()

        # Use the harder of escalated vs ALP-selected difficulty
        _rank = {"easy": 0, "medium": 1, "hard": 2}
        chosen_difficulty = (
            alp_difficulty if _rank.get(alp_difficulty, 0) >= _rank.get(escalated, 0)
            else escalated
        )

        try:
            chosen_failure_type = FailureType(alp_failure_type_str)
        except ValueError:
            chosen_failure_type = FailureType.cpu_spike

        # Select root cause service — pick from services NOT yet in blast radius
        # (gap-aware: avoids repeating the same service that was just failing)
        all_svcs = list(ALL_SERVICES)
        last_root = source_template.root_cause_service if source_template else ""
        candidate_svcs = [s for s in all_svcs if s != last_root] or all_svcs
        # Use gap string as a deterministic-but-varied seed
        root_service = candidate_svcs[hash(gap + chosen_failure_type.value) % len(candidate_svcs)]

        # Build signals based on difficulty and gap
        ground_truth, red_herrings = _build_signals(
            root_service, chosen_failure_type, chosen_difficulty, gap
        )
        missing_log_ratio = {"easy": 0.05, "medium": 0.25, "hard": 0.40}[chosen_difficulty]
        cascade_risk = {"easy": "low", "medium": "medium", "hard": "high"}[chosen_difficulty]
        step_range = {"easy": (5, 20), "medium": (20, 60), "hard": (60, 150)}[chosen_difficulty]

        template_id = f"ORACLE-{uuid.uuid4().hex[:8].upper()}"
        new_template = IncidentTemplate(
            id=template_id,
            name=f"ORACLE-ALP: {gap} gap | {chosen_failure_type.value} ({chosen_difficulty})",
            difficulty=chosen_difficulty,  # type: ignore[arg-type]
            root_cause_service=root_service,
            failure_type=chosen_failure_type,
            ground_truth_signals=ground_truth,
            red_herring_signals=red_herrings,
            cascade_risk=cascade_risk,  # type: ignore[arg-type]
            missing_log_ratio=missing_log_ratio,
            expected_steps_to_resolve=step_range,
        )

        logger.info(
            "ORACLE: ALP selected (%s, %s) gap=%s — root=%s",
            chosen_difficulty, chosen_failure_type.value, gap, root_service,
        )

        # Validate and add to library (Req 12.5)
        if self.incident_generator is not None:
            try:
                self.incident_generator.validate_template(new_template)
                self.incident_generator.add_template(new_template)
            except ValueError:
                return new_template

        self.oracle_template_count += 1
        self.oracle_template_utility[template_id] = 0.5
        self._retire_if_needed()

        return new_template

    # ------------------------------------------------------------------
    # Template retirement (Req 12.7)
    # ------------------------------------------------------------------

    def _retire_if_needed(self) -> None:
        """Retire oldest below-median utility templates when count > 50."""
        if self.oracle_template_count <= 50:
            return

        if not self.oracle_template_utility:
            return

        utilities = list(self.oracle_template_utility.values())
        if len(utilities) < 2:
            return

        median_utility = statistics.median(utilities)

        # Find below-median templates
        below_median = [
            tid for tid, util in self.oracle_template_utility.items()
            if util < median_utility
        ]

        if not below_median:
            return

        # Retire the oldest (first inserted) below-median template
        # Python dicts preserve insertion order (3.7+)
        to_retire = below_median[0]
        del self.oracle_template_utility[to_retire]
        self.oracle_template_count -= 1

        # Remove from incident_generator if available
        if self.incident_generator is not None:
            self.incident_generator._templates = [
                t for t in self.incident_generator._templates
                if t.id != to_retire
            ]

    def retire_below_median_templates(self) -> list[str]:
        """Explicitly retire all below-median templates when count > 50.

        Returns the list of retired template IDs.
        """
        if self.oracle_template_count <= 50:
            return []

        if not self.oracle_template_utility:
            return []

        utilities = list(self.oracle_template_utility.values())
        if len(utilities) < 2:
            return []

        median_utility = statistics.median(utilities)

        below_median = [
            tid for tid, util in self.oracle_template_utility.items()
            if util < median_utility
        ]

        retired: list[str] = []
        for tid in below_median:
            del self.oracle_template_utility[tid]
            self.oracle_template_count -= 1
            retired.append(tid)

            if self.incident_generator is not None:
                self.incident_generator._templates = [
                    t for t in self.incident_generator._templates
                    if t.id != tid
                ]

        return retired

    def set_template_utility(self, template_id: str, utility: float) -> None:
        """Update the utility score for a tracked template."""
        if template_id in self.oracle_template_utility:
            self.oracle_template_utility[template_id] = utility


# ---------------------------------------------------------------------------
# Signal builder (module-level helper)
# ---------------------------------------------------------------------------

def _build_signals(
    root_service: str,
    failure_type: "FailureType",
    difficulty: str,
    gap: str,
) -> tuple[list[str], list[str]]:
    """Build ground-truth and red-herring signal lists for a generated template.

    Signal richness scales with difficulty:
      easy  — 3 clear signals, 0 red herrings
      medium— 3 signals, 2 red herrings
      hard  — 2 signals, 4 red herrings (adversarial)
    """
    ft = failure_type.value
    ground_truth = [
        f"{root_service}::{ft}_detected",
        f"{root_service}::error_rate_spike",
        f"{root_service}::latency_p99_breach",
    ]
    if difficulty == "hard":
        ground_truth = ground_truth[:2]  # hide one signal on hard

    _rh_pool = [
        "adjacent_service::spurious_latency",
        "load_balancer::connection_timeout",
        "cache::miss_rate_noise",
        "db::slow_query_noise",
        "mesh::retry_storm_artifact",
    ]
    n_rh = {"easy": 0, "medium": 2, "hard": 4}[difficulty]
    # Deterministic shuffle from gap seed so reproducible per gap
    rng = random.Random(hash(gap + root_service))
    rh_pool = list(_rh_pool)
    rng.shuffle(rh_pool)
    red_herrings = rh_pool[:n_rh]

    return ground_truth, red_herrings
