"""ORACLE — Self-Improvement Agent for SENTINEL.

Analyzes completed incident trajectories to identify capability gaps,
stores trajectories and embeddings in ChromaDB (falls back to in-memory),
generates new IncidentTemplates targeting identified gaps,
and retires below-median templates when library exceeds 50 ORACLE-generated entries.

Scenario generation uses GPT-4o-mini via OpenAI API when OPENAI_API_KEY is set;
falls back to a config-driven heuristic when the key is absent.

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
"""
from __future__ import annotations

import logging
import statistics
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING

from sentinel.agents.base import BaseAgent
from sentinel.llm_client import get_openai_client
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
        """Generate a new IncidentTemplate targeting the identified capability gap.

        When OPENAI_API_KEY is set, calls GPT-4o-mini to produce scenario params.
        Falls back to a heuristic template when the API is unavailable.

        Assigns difficulty one level above the source incident (capped at Hard).
        Submits to Incident_Generator for validation before adding to library.
        """
        gap = self.analyze_trajectory(trajectory)

        # Determine difficulty escalation
        source_difficulty = source_template.difficulty if source_template else "easy"
        new_difficulty = _DIFFICULTY_ESCALATION.get(source_difficulty, "hard")

        template_id = f"ORACLE-{uuid.uuid4().hex[:8].upper()}"

        # --- Try GPT-4o-mini first ---
        all_services = list(ALL_SERVICES)
        failure_type_values = [ft.value for ft in FailureType]

        oai = get_openai_client()
        llm_params: dict | None = None
        if oai is not None:
            llm_params = oai.generate_incident_params(
                capability_gap=gap,
                difficulty=new_difficulty,
                all_services=all_services,
                failure_types=failure_type_values,
            )
            if llm_params:
                logger.info(
                    "ORACLE: GPT-4o-mini generated scenario params for gap=%s", gap
                )

        # --- Parse LLM response or fall back to heuristic ---
        if llm_params:
            # Validate and coerce LLM output
            root_cause = llm_params.get("root_cause_service", "")
            if root_cause not in all_services:
                root_cause = all_services[0]

            ft_value = llm_params.get("failure_type", FailureType.cpu_spike.value)
            try:
                failure_type = FailureType(ft_value)
            except ValueError:
                failure_type = FailureType.cpu_spike

            ground_truth = llm_params.get(
                "ground_truth_signals",
                [f"{gap}_anomaly_signal"],
            )
            red_herrings = llm_params.get("red_herring_signals", [])
            cascade_risk = llm_params.get("cascade_risk", "medium")
            if cascade_risk not in ("low", "medium", "high"):
                cascade_risk = "medium"
            missing_log_ratio = float(llm_params.get("missing_log_ratio", 0.3))
            missing_log_ratio = max(0.0, min(0.5, missing_log_ratio))
            steps = llm_params.get("expected_steps_to_resolve", [10, 50])
            expected_steps: tuple[int, int] = (
                int(steps[0]) if isinstance(steps, list) and len(steps) >= 2 else 10,
                int(steps[1]) if isinstance(steps, list) and len(steps) >= 2 else 50,
            )
        else:
            # Heuristic fallback — use gap to select a varied service/failure type
            _gap_failure_map = {
                "investigative": FailureType.connection_pool_exhaustion,
                "remediation": FailureType.memory_leak,
                "deployment": FailureType.bad_deployment,
                "meta": FailureType.network_partition,
            }
            failure_type = _gap_failure_map.get(gap, FailureType.cpu_spike)
            root_cause = all_services[hash(gap) % len(all_services)]
            ground_truth = [f"{gap}_anomaly_signal", f"{root_cause}_error_spike"]
            red_herrings = ["spurious_alert_1"]
            cascade_risk = "medium"
            missing_log_ratio = 0.3
            expected_steps = (10, 50)

        new_template = IncidentTemplate(
            id=template_id,
            name=f"ORACLE-generated: {gap} gap ({new_difficulty})",
            difficulty=new_difficulty,  # type: ignore[arg-type]
            root_cause_service=root_cause,
            failure_type=failure_type,
            ground_truth_signals=ground_truth,
            red_herring_signals=red_herrings,
            cascade_risk=cascade_risk,  # type: ignore[arg-type]
            missing_log_ratio=missing_log_ratio,
            expected_steps_to_resolve=expected_steps,
        )

        # Validate and add to library (Req 12.5)
        if self.incident_generator is not None:
            try:
                self.incident_generator.validate_template(new_template)
                self.incident_generator.add_template(new_template)
            except ValueError:
                # Discard invalid template (per error handling spec)
                return new_template

        # Track ORACLE-generated template
        self.oracle_template_count += 1
        self.oracle_template_utility[template_id] = 0.5  # initial utility score

        # Retire below-median templates if library exceeds 50 (Req 12.7)
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
