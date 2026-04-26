"""Reward function for SENTINEL.

Implements the 4-dimensional RLVR reward signal:
  R1 — Root cause accuracy (0.0, 0.5, or 1.0)
  R2 — MTTR score (inversely proportional, with pre-SLA bonus)
  R3 — Recovery quality (fraction of services within 5% of baseline)
  R4 — Blast radius minimization (1 - final_br / peak_br)

Step rewards apply hard penalties for blast radius expansion and
unnecessary restarts of healthy services.

Episode reward = 0.35*R1 + 0.30*R2 + 0.25*R3 + 0.10*R4 + penalties
"""
from __future__ import annotations

from sentinel.models import (
    Action,
    RewardBreakdown,
    RewardWeights,
    Trajectory,
)
from sentinel.world_state import NexaStackWorldState
from sentinel.models import IncidentState

# ---------------------------------------------------------------------------
# Baseline values used for R3 recovery quality check
# ---------------------------------------------------------------------------

_BASELINE_METRICS: dict[str, float] = {
    "cpu": 0.2,
    "memory": 0.3,
    "latency_ms": 50.0,
    "error_rate": 0.01,
    "saturation": 0.3,
}

_RECOVERY_TOLERANCE = 0.05  # 5% of baseline value


class Reward_Function:
    """Computes step and episode rewards for SENTINEL."""

    def __init__(self, weights: RewardWeights, sla_breach_threshold: int) -> None:
        self.w = weights
        self.sla_breach_threshold = sla_breach_threshold

    # ------------------------------------------------------------------
    # Step reward
    # ------------------------------------------------------------------

    def compute_step_reward(
        self,
        action: Action,
        world_state: NexaStackWorldState,
        incident_state: IncidentState,
        previous_blast_radius: set[str] | None = None,
        current_blast_radius: set[str] | None = None,
        target_was_healthy: bool = False,
    ) -> float:
        """Return the immediate step reward for *action*.

        Reward shaping provides dense signal beyond just penalties:
        - Investigative actions targeting the actual root cause service: +0.15
        - FormHypothesis correctly identifying root cause service: +0.20
        - FormHypothesis correctly identifying service + failure type: +0.30
        - Remediation targeting the actual root cause: +0.25
        - Remediation that reduces blast radius: +0.10 per service recovered
        - Wrong-service investigative actions: -0.05 (mild discouragement)
        - Blast radius expansion: -1.0
        - Healthy-service restart: -1.0
        """
        reward = 0.0

        # ── Blast radius expansion penalty ────────────────────────────
        pre_br = set(previous_blast_radius or incident_state.current_blast_radius)
        post_br = set(
            current_blast_radius
            or (
                world_state.incident_state.current_blast_radius
                if world_state.incident_state is not None
                else pre_br
            )
        )
        if len(post_br) > len(pre_br):
            reward -= 1.0

        # ── Blast radius reduction bonus ──────────────────────────────
        if len(post_br) < len(pre_br):
            services_recovered = len(pre_br) - len(post_br)
            reward += 0.10 * services_recovered

        # ── Healthy-service restart penalty ───────────────────────────
        if action.name == "RestartService":
            target = action.params.get("service", "")
            if target and target in world_state.services:
                if not target_was_healthy:
                    target_was_healthy = world_state.services[target].availability
            if target and target_was_healthy:
                reward -= 1.0

        # ── Investigative reward shaping ──────────────────────────────
        target_service = action.params.get("service", "")
        root_cause = incident_state.root_cause_service

        if action.category == "investigative" and target_service:
            if target_service == root_cause:
                reward += 0.15   # investigating the right service
            else:
                reward -= 0.05   # mild penalty for investigating wrong service

        # ── FormHypothesis shaping ────────────────────────────────────
        if action.name == "FormHypothesis":
            hyp_service = action.params.get("service", "")
            hyp_ft = action.params.get("failure_type", "")
            if hyp_service == root_cause:
                if hyp_ft == incident_state.failure_type.value:
                    reward += 0.30   # correct service + failure type
                else:
                    reward += 0.20   # correct service, wrong failure type
            else:
                reward -= 0.10   # wrong hypothesis

        # ── Targeted remediation shaping ──────────────────────────────
        if action.category == "remediation" and target_service:
            if target_service == root_cause:
                reward += 0.25   # remediating the correct root cause
            elif target_service in incident_state.current_blast_radius:
                reward += 0.05   # remediating an affected (but not root) service

        return reward

    # ------------------------------------------------------------------
    # Episode reward
    # ------------------------------------------------------------------

    def compute_episode_reward(
        self,
        trajectory: Trajectory,
        world_state: NexaStackWorldState,
        incident_state: IncidentState,
    ) -> RewardBreakdown:
        """Compute the full episode reward and return a RewardBreakdown.

        Requires the trajectory to carry the identified root cause via
        ``trajectory.steps[-1].info`` keys ``identified_root_cause`` and
        ``identified_failure_type`` (both default to empty string if absent).
        """
        # Extract identified root cause from the last step's info dict.
        # If the caller did not explicitly attach this, infer it from the
        # trajectory / incident state so episode rewards remain meaningful.
        last_info: dict = trajectory.steps[-1].info if trajectory.steps else {}
        identified_root_cause: str = last_info.get("identified_root_cause", "")
        identified_failure_type: str = last_info.get("identified_failure_type", "")
        if not identified_root_cause or not identified_failure_type:
            inferred_service, inferred_type = self._infer_identification(
                trajectory, incident_state
            )
            identified_root_cause = identified_root_cause or inferred_service
            identified_failure_type = identified_failure_type or inferred_type

        r1 = self._r1_root_cause_accuracy(
            incident_state, identified_root_cause, identified_failure_type
        )
        r2 = self._r2_mttr(trajectory.mttr, r1)
        r3 = self._r3_recovery_quality(world_state)
        r4 = self._r4_blast_radius(incident_state)

        # Late resolution penalty
        penalties = 0.0
        if trajectory.mttr > 2 * self.sla_breach_threshold:
            penalties -= 0.5

        total = (
            self.w.r1_root_cause * r1
            + self.w.r2_mttr * r2
            + self.w.r3_recovery_quality * r3
            + self.w.r4_blast_radius * r4
            + penalties
        )

        return RewardBreakdown(
            r1=r1,
            r2=r2,
            r3=r3,
            r4=r4,
            penalties=penalties,
            total=total,
        )

    # ------------------------------------------------------------------
    # Component reward helpers
    # ------------------------------------------------------------------

    def _r1_root_cause_accuracy(
        self,
        incident_state: IncidentState,
        identified_root_cause: str,
        identified_failure_type: str,
    ) -> float:
        """Return 1.0, 0.5, or 0.0 based on root cause identification accuracy.

        - 1.0: both service AND failure_type match ground truth
        - 0.5: only service matches
        - 0.0: service does not match
        """
        service_match = identified_root_cause == incident_state.root_cause_service
        type_match = identified_failure_type == incident_state.failure_type.value

        if service_match and type_match:
            return 1.0
        if service_match:
            return 0.5
        return 0.0

    def _r2_mttr(self, mttr_steps: int, r1: float = 0.0) -> float:
        """Return an MTTR score inversely proportional to resolution time.

        Score = 1.0 / (1.0 + mttr_steps / sla_breach_threshold)
        +0.1 bonus if mttr_steps < sla_breach_threshold (pre-SLA resolution)
        Clamped to [0.0, 1.1].

        If r1 == 0 (no root cause identified), MTTR score is halved
        to discourage episodes that run to max_steps without diagnosis.
        """
        base = 1.0 / (1.0 + mttr_steps / self.sla_breach_threshold)
        bonus = 0.1 if mttr_steps < self.sla_breach_threshold else 0.0
        score = max(0.0, min(1.1, base + bonus))
        if r1 == 0.0:
            score *= 0.5  # Penalize MTTR when root cause not identified
        return score

    def _r3_recovery_quality(self, world_state: NexaStackWorldState) -> float:
        """Return the fraction of services whose metrics are within 5% of baseline.

        A service is "recovered" when every metric is within 5% of its
        baseline value (relative tolerance for all metrics).
        """
        recovered = 0
        total = len(world_state.services)

        for metrics in world_state.services.values():
            if self._is_recovered(metrics):
                recovered += 1

        return recovered / total if total > 0 else 1.0

    def _r4_blast_radius(self, incident_state: IncidentState) -> float:
        """Return 1.0 - (current_blast_radius / peak_blast_radius).

        Returns 1.0 when peak_blast_radius is empty (no blast occurred).
        """
        peak = len(incident_state.peak_blast_radius)
        if peak == 0:
            return 1.0
        current = len(incident_state.current_blast_radius)
        return 1.0 - (current / peak)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_recovered(self, metrics) -> bool:
        """Return True if all metrics are within 5% of their baseline values."""
        checks = [
            ("cpu", metrics.cpu),
            ("memory", metrics.memory),
            ("latency_ms", metrics.latency_ms),
            ("error_rate", metrics.error_rate),
            ("saturation", metrics.saturation),
        ]
        for name, value in checks:
            baseline = _BASELINE_METRICS[name]
            if baseline == 0.0:
                if abs(value) > _RECOVERY_TOLERANCE:
                    return False
            else:
                if abs(value - baseline) / baseline > _RECOVERY_TOLERANCE:
                    return False
        return True

    def _infer_identification(
        self,
        trajectory: Trajectory,
        incident_state: IncidentState,
    ) -> tuple[str, str]:
        """Infer the final diagnosis from explicit metadata or hypotheses."""
        if incident_state.active_hypotheses:
            best = max(
                incident_state.active_hypotheses,
                key=lambda h: h.confidence,
            )
            return best.service, best.failure_type.value

        for step in reversed(trajectory.steps):
            if step.action.name == "FormHypothesis":
                return (
                    step.action.params.get("service", ""),
                    step.action.params.get("failure_type", ""),
                )

        # For remediation agents (Forge): infer root cause from which service
        # was remediated most — gives Forge partial R1 credit
        remediation_targets: dict[str, int] = {}
        for step in trajectory.steps:
            if step.action.category == "remediation":
                svc = step.action.params.get("service", "")
                if svc:
                    remediation_targets[svc] = remediation_targets.get(svc, 0) + 1
        if remediation_targets:
            most_remediated = max(remediation_targets, key=remediation_targets.get)
            return most_remediated, ""

        return "", ""
