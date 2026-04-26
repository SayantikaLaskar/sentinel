"""Sentinel_Env — top-level Gymnasium environment for SENTINEL.

Wires together NexaStackWorldState, Observability_Layer, Incident_Generator,
Reward_Function, and all agents into a single gym.Env-compatible interface.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import gymnasium
import gymnasium.spaces

from sentinel.cascade_engine import Cascade_Engine
from sentinel.config import load_config
from sentinel.exceptions import IncidentLibraryError
from sentinel.incident_generator import Incident_Generator
from sentinel.models import (
    Action,
    AgentMessage,
    IncidentState,
    MessageBus,
    RewardWeights,
)
from sentinel.observability import Observability_Layer
from sentinel.reward import Reward_Function
from sentinel.world_state import ALL_SERVICES, NexaStackWorldState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent role constraints
# ---------------------------------------------------------------------------

_AGENT_ALLOWED_CATEGORIES: dict[str, list[str]] = {
    "argus":  ["investigative", "meta"],
    "holmes": ["investigative"],
    "forge":  ["remediation"],
    "hermes": ["deployment", "meta"],
    "oracle": ["meta"],
}


class Sentinel_Env(gymnasium.Env):
    """Multi-agent incident response RL environment for NexaStack."""

    metadata = {"render_modes": ["human", "json"]}

    def __init__(
        self,
        config_path: str = "env_spec.yaml",
        incident_library_path: str = "incident_library.yaml",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode

        # Load config (falls back to defaults with warning if absent)
        self._config = load_config(config_path)

        # Incident generator — raises IncidentLibraryError if library is missing/malformed
        self.incident_generator = Incident_Generator(incident_library_path)

        # World state and supporting components
        self.world_state = NexaStackWorldState()
        self._cascade_engine = Cascade_Engine()

        obs_cfg = self._config.observability
        self.observability_layer = Observability_Layer(obs_cfg)

        rw_cfg = self._config.reward.weights
        weights = RewardWeights(
            r1_root_cause=rw_cfg.r1_root_cause,
            r2_mttr=rw_cfg.r2_mttr,
            r3_recovery_quality=rw_cfg.r3_recovery_quality,
            r4_blast_radius=rw_cfg.r4_blast_radius,
        )
        self.reward_function = Reward_Function(
            weights=weights,
            sla_breach_threshold=self._config.training.sla_breach_threshold_steps,
        )

        self._max_steps: int = self._config.training.max_steps_per_episode
        self._difficulty_distribution: dict[str, float] = {
            "easy":   self._config.incident.difficulty_distribution.easy,
            "medium": self._config.incident.difficulty_distribution.medium,
            "hard":   self._config.incident.difficulty_distribution.hard,
        }

        # Episode state
        self.current_episode: dict | None = None
        self._incident_state: IncidentState | None = None
        self.step_count: int = 0
        self._needs_reset: bool = True
        self._episode_id: str = ""

        # Multi-agent communication bus
        self.message_bus = MessageBus()

        # Define spaces
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    # ------------------------------------------------------------------
    # Space definitions
    # ------------------------------------------------------------------

    def _build_observation_space(self) -> gymnasium.spaces.Dict:
        import numpy as np

        n = len(ALL_SERVICES)
        return gymnasium.spaces.Dict({
            # Flat JSON string for the metrics snapshot dict
            "metrics_snapshot": gymnasium.spaces.Text(min_length=0, max_length=65536),
            # 30×30 adjacency matrix flattened
            "causal_graph_snapshot": gymnasium.spaces.Box(
                low=0.0, high=1.0, shape=(n * n,), dtype=np.float32
            ),
            # JSON-encoded list of alerts
            "active_alerts": gymnasium.spaces.Text(min_length=0, max_length=65536),
            # JSON-encoded list of log entries
            "recent_logs": gymnasium.spaces.Text(min_length=0, max_length=65536),
            # JSON-encoded list of traces
            "active_traces": gymnasium.spaces.Text(min_length=0, max_length=65536),
            # JSON-encoded incident context dict
            "incident_context": gymnasium.spaces.Text(min_length=0, max_length=65536),
            # JSON-encoded sla_state dict
            "sla_state": gymnasium.spaces.Text(min_length=0, max_length=65536),
        })

    def _build_action_space(self) -> gymnasium.spaces.Dict:
        return gymnasium.spaces.Dict({
            "agent":    gymnasium.spaces.Text(min_length=1, max_length=64),
            "category": gymnasium.spaces.Text(min_length=1, max_length=64),
            "name":     gymnasium.spaces.Text(min_length=1, max_length=128),
            # JSON-encoded params dict
            "params":   gymnasium.spaces.Text(min_length=0, max_length=65536),
        })

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset the environment and inject a new incident.

        Returns:
            (observation, info) where info contains the incident_id.
        """
        super().reset(seed=seed)

        # 1. Restore world state to baseline
        self.world_state.restore_baseline()
        self.step_count = 0
        self._needs_reset = False
        self.message_bus.clear()
        self._episode_id = str(uuid.uuid4())

        # 2. Sample a new incident template
        template = self.incident_generator.sample(self._difficulty_distribution)

        # 3. Create IncidentState from template
        incident_state = IncidentState(
            template_id=template.id,
            root_cause_service=template.root_cause_service,
            failure_type=template.failure_type,
            ground_truth_signals=list(template.ground_truth_signals),
            red_herring_signals=list(template.red_herring_signals),
            affected_services={},
            peak_blast_radius=set(),
            current_blast_radius=set(),
            timeline=[],
            attempted_remediations=[],
            active_hypotheses=[],
            resolved=False,
            step_injected=0,
        )

        # 4. Inject failure via Cascade_Engine
        affected = self._cascade_engine.propagate_failure(
            world_state=self.world_state,
            root_service=template.root_cause_service,
            failure_type=template.failure_type,
            initial_severity=1.0,
        )
        blast_radius = set(affected.keys())
        incident_state.affected_services = affected
        incident_state.peak_blast_radius = set(blast_radius)
        incident_state.current_blast_radius = set(blast_radius)

        # Attach incident state to world state
        self.world_state.incident_state = incident_state
        self._incident_state = incident_state

        # 5. Sample episode params for Observability_Layer
        self.observability_layer.sample_episode_params()

        # 6. Build and return initial observation
        obs = self._build_obs()
        info = {"incident_id": template.id}
        return obs, info

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        """Apply an action and advance the environment by one step.

        Args:
            action: dict with keys agent, category, name, params.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # 1. Raise ResetNeeded if not reset yet
        if self._needs_reset:
            raise gymnasium.error.ResetNeeded(
                "Call reset() before step()."
            )

        # 2. Parse action dict into Action model
        if isinstance(action, dict):
            # params may be a JSON string (from action_space) or a dict
            params = action.get("params", {})
            if isinstance(params, str):
                try:
                    params = json.loads(params) if params else {}
                except json.JSONDecodeError:
                    params = {}
            try:
                parsed_action = Action(
                    agent=action["agent"],
                    category=action["category"],
                    name=action["name"],
                    params=params,
                )
            except Exception:
                obs = self._build_obs()
                return obs, -0.1, False, False, {"error": "invalid_action"}
        else:
            parsed_action = action

        # 3. Validate action
        # Role violation check
        allowed_categories = _AGENT_ALLOWED_CATEGORIES.get(parsed_action.agent, [])
        if parsed_action.category not in allowed_categories:
            obs = self._build_obs()
            return obs, -0.1, False, False, {"error": "role_violation"}

        # Unknown service check
        service_param = parsed_action.params.get("service", None)
        if service_param is not None and service_param not in self.world_state.services:
            obs = self._build_obs()
            return obs, -0.1, False, False, {"error": "unknown_service"}

        previous_blast_radius = set(
            self._incident_state.current_blast_radius
        ) if self._incident_state is not None else set()
        service_param = parsed_action.params.get("service")
        target_was_healthy = bool(
            service_param in self.world_state.services
            and self.world_state.services[service_param].availability
        ) if service_param else False

        # 4. Apply action effects to world state
        self._apply_action(parsed_action)

        # 4b. Generate inter-agent messages from action results
        self._generate_agent_messages(parsed_action)

        # 5. Compute step reward
        incident_state = self._incident_state
        reward = self.reward_function.compute_step_reward(
            action=parsed_action,
            world_state=self.world_state,
            incident_state=incident_state,
            previous_blast_radius=previous_blast_radius,
            current_blast_radius=(
                set(incident_state.current_blast_radius)
                if incident_state is not None else set()
            ),
            target_was_healthy=target_was_healthy,
        )

        # 6. Increment step count
        self.step_count += 1
        self.world_state.step = self.step_count

        # 6b. Dynamic cascade: unresolved failures spread every 5 steps
        if self.step_count % 5 == 0 and incident_state is not None:
            self._propagate_secondary_failures()

        # 7. Check termination
        # FormHypothesis ends episode for investigative agents (Holmes/Argus)
        hypothesis_terminates = (
            parsed_action.name == "FormHypothesis"
            and parsed_action.agent in ("holmes", "argus")
        )
        # Blast radius reaching 0 ends episode for remediation agents (Forge)
        blast_resolved = (
            incident_state is not None
            and parsed_action.category == "remediation"
            and len(incident_state.current_blast_radius) == 0
        )
        # Deployment success ends episode for Hermes (FullDeploy/Rollback on root cause)
        deploy_resolved = (
            parsed_action.agent == "hermes"
            and parsed_action.name in ("FullDeploy", "Rollback")
            and incident_state is not None
            and len(incident_state.current_blast_radius) == 0
        )
        # EscalateToHuman ends episode for Oracle
        oracle_escalated = (
            parsed_action.agent == "oracle"
            and parsed_action.name == "EscalateToHuman"
        )
        terminated = (
            parsed_action.name == "CloseIncident"
            or hypothesis_terminates
            or blast_resolved
            or deploy_resolved
            or oracle_escalated
            or self.step_count >= self._max_steps
        )
        truncated = False

        if terminated:
            self._needs_reset = True
            if incident_state is not None:
                incident_state.resolved = (
                    parsed_action.name == "CloseIncident"
                    or hypothesis_terminates
                    or blast_resolved
                    or deploy_resolved
                )

        # 8. Build observation
        obs = self._build_obs()
        info: dict[str, Any] = {
            "step_count": self.step_count,
            "episode_id": self._episode_id,
            "current_blast_radius": (
                sorted(incident_state.current_blast_radius)
                if incident_state is not None else []
            ),
        }
        if incident_state is not None:
            identified_root_cause, identified_failure_type = self._get_best_identification()
            info["identified_root_cause"] = identified_root_cause
            info["identified_failure_type"] = identified_failure_type
        if terminated:
            info["terminated_reason"] = (
                "CloseIncident" if parsed_action.name == "CloseIncident"
                else "FormHypothesis" if hypothesis_terminates
                else "blast_resolved" if blast_resolved
                else "deploy_resolved" if deploy_resolved
                else "oracle_escalated" if oracle_escalated
                else "max_steps"
            )

        return obs, float(reward), terminated, truncated, info

    def render(self) -> str | None:
        """Render the current environment state.

        Returns a human-readable string or JSON string depending on render_mode.
        """
        if self.render_mode is None:
            return None

        snapshot = self.world_state.snapshot()

        if self.render_mode == "json":
            return json.dumps(snapshot, indent=2)

        # human mode — compact text summary
        lines = [f"=== SENTINEL Episode {self._episode_id} | Step {self.step_count} ==="]
        if self._incident_state is not None:
            inc = self._incident_state
            lines.append(
                f"Incident: {inc.template_id} | Root: {inc.root_cause_service} "
                f"({inc.failure_type.value})"
            )
            lines.append(
                f"Blast radius: {len(inc.current_blast_radius)} services "
                f"(peak: {len(inc.peak_blast_radius)})"
            )
        degraded = [
            svc for svc, m in self.world_state.services.items() if not m.availability
        ]
        lines.append(f"Degraded services ({len(degraded)}): {', '.join(degraded) or 'none'}")
        return "\n".join(lines)

    def close(self) -> None:
        """Clean up environment resources."""
        self._needs_reset = True
        self._incident_state = None
        self.world_state.incident_state = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict:
        """Build a gymnasium-compatible observation dict."""
        import numpy as np

        raw = self.observability_layer.build_observation(
            world_state=self.world_state,
            incident_state=self._incident_state,
            hypothesis_tree=None,
            step_count=self.step_count,
        )

        # Flatten causal_graph_snapshot (list[list[float]]) to 1-D np.float32 array
        matrix = raw["causal_graph_snapshot"]
        flat_matrix = np.array(
            [cell for row in matrix for cell in row], dtype=np.float32
        )

        # Serialize complex fields to JSON strings for the Text spaces
        def _serialize(obj: Any) -> str:
            if hasattr(obj, "model_dump"):
                return json.dumps(obj.model_dump())
            if isinstance(obj, list):
                items = []
                for item in obj:
                    if hasattr(item, "model_dump"):
                        items.append(item.model_dump())
                    else:
                        items.append(item)
                return json.dumps(items)
            return json.dumps(obj)

        return {
            "metrics_snapshot": _serialize(raw["metrics_snapshot"]),
            "causal_graph_snapshot": flat_matrix,
            "active_alerts": _serialize(raw["active_alerts"]),
            "recent_logs": _serialize(raw["recent_logs"]),
            "active_traces": _serialize(raw["active_traces"]),
            "incident_context": _serialize(raw["incident_context"]),
            "sla_state": _serialize(raw["sla_state"]),
        }

    def _apply_action(self, action: Action) -> None:
        """Apply the action's effects to the world state."""
        name = action.name
        params = action.params
        incident_state = self._incident_state

        # ── Investigative actions ────────────────────────────────────────
        if name == "QueryLogs":
            service = params.get("service", "")
            if service and incident_state is not None:
                from sentinel.models import TimelineEntry
                incident_state.timeline.append(
                    TimelineEntry(
                        step=self.step_count,
                        event_type="query_logs",
                        description=f"Queried logs for {service}",
                        agent=action.agent,
                    )
                )

        elif name == "QueryMetrics":
            service = params.get("service", "")
            if service and incident_state is not None:
                from sentinel.models import TimelineEntry
                incident_state.timeline.append(
                    TimelineEntry(
                        step=self.step_count,
                        event_type="query_metrics",
                        description=f"Queried metrics for {service}: {params.get('metric_name', 'all')}",
                        agent=action.agent,
                    )
                )

        elif name == "QueryTrace":
            if incident_state is not None:
                from sentinel.models import TimelineEntry
                incident_state.timeline.append(
                    TimelineEntry(
                        step=self.step_count,
                        event_type="query_trace",
                        description=f"Queried trace {params.get('trace_id', 'unknown')}",
                        agent=action.agent,
                    )
                )

        elif name == "FormHypothesis":
            service = params.get("service", "")
            failure_type = params.get("failure_type", "")
            confidence = params.get("confidence", 0.5)
            # Robust float conversion — LLM may output "high", "0.9", etc.
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
                confidence = confidence_map.get(str(confidence).lower(), 0.5)
            if service and incident_state is not None:
                from sentinel.models import HypothesisNode, FailureType, TimelineEntry
                # Normalize failure_type — LLM may generate variants
                ft_str = failure_type.lower().replace("-", "_").replace(" ", "_")
                try:
                    ft = FailureType(ft_str)
                except ValueError:
                    # Try partial match
                    ft = FailureType.cpu_spike
                    for member in FailureType:
                        if member.value in ft_str or ft_str in member.value:
                            ft = member
                            break
                node = HypothesisNode(
                    service=service, failure_type=ft, confidence=float(confidence)
                )
                # Replace existing hypothesis for same service or add new
                incident_state.active_hypotheses = [
                    h for h in incident_state.active_hypotheses if h.service != service
                ]
                incident_state.active_hypotheses.append(node)
                incident_state.timeline.append(
                    TimelineEntry(
                        step=self.step_count,
                        event_type="form_hypothesis",
                        description=f"Hypothesis: {service} ({failure_type}) conf={confidence:.2f}",
                        agent=action.agent,
                    )
                )

        # ── Remediation actions ──────────────────────────────────────────
        elif name == "RestartService":
            service = params.get("service", "")
            if service and service in self.world_state.services:
                self._cascade_engine.propagate_recovery(
                    world_state=self.world_state,
                    resolved_service=service,
                )
                if incident_state is not None:
                    new_br = self._cascade_engine.get_blast_radius()
                    incident_state.current_blast_radius = new_br

        elif name == "ScaleService":
            service = params.get("service", "")
            replicas = params.get("replicas", 2)
            if service and service in self.world_state.services:
                m = self.world_state.services[service]
                # Scaling reduces saturation and CPU proportionally
                scale_factor = max(0.3, 1.0 / max(replicas, 1))
                from sentinel.models import ServiceMetrics
                self.world_state.services[service] = ServiceMetrics(
                    cpu=max(0.1, m.cpu * scale_factor),
                    memory=m.memory,  # memory stays (per-instance)
                    latency_ms=max(10.0, m.latency_ms * scale_factor),
                    error_rate=max(0.0, m.error_rate * scale_factor),
                    saturation=max(0.05, m.saturation * scale_factor),
                    availability=True if m.saturation * scale_factor < 0.9 else m.availability,
                )
                if incident_state is not None:
                    # May reduce blast radius if service is now healthy
                    if self.world_state.services[service].availability:
                        incident_state.current_blast_radius.discard(service)

        elif name == "RollbackDeployment":
            service = params.get("service", "")
            if service and service in self.world_state.services:
                # Rollback is highly effective if failure is bad_deployment
                if incident_state is not None and incident_state.failure_type.value == "bad_deployment":
                    if service == incident_state.root_cause_service:
                        self._cascade_engine.propagate_recovery(
                            world_state=self.world_state,
                            resolved_service=service,
                        )
                        incident_state.current_blast_radius = self._cascade_engine.get_blast_radius()
                    else:
                        # Partial effect on non-root service
                        from sentinel.world_state import _baseline_metrics
                        self.world_state.services[service] = _baseline_metrics()
                        incident_state.current_blast_radius.discard(service)

        elif name == "DrainTraffic":
            service = params.get("service", "")
            if service and service in self.world_state.services:
                m = self.world_state.services[service]
                from sentinel.models import ServiceMetrics
                # Draining traffic reduces error rate and latency but doesn't fix root cause
                self.world_state.services[service] = ServiceMetrics(
                    cpu=m.cpu,
                    memory=m.memory,
                    latency_ms=max(10.0, m.latency_ms * 0.3),
                    error_rate=max(0.0, m.error_rate * 0.2),
                    saturation=max(0.05, m.saturation * 0.3),
                    availability=True,
                )
                if incident_state is not None:
                    incident_state.current_blast_radius.discard(service)

        elif name == "ModifyRateLimit":
            service = params.get("service", "")
            limit_rps = params.get("limit_rps", 100)
            if service and service in self.world_state.services:
                m = self.world_state.services[service]
                # Rate limiting caps saturation
                cap = min(1.0, limit_rps / 200.0)
                from sentinel.models import ServiceMetrics
                self.world_state.services[service] = ServiceMetrics(
                    cpu=m.cpu,
                    memory=m.memory,
                    latency_ms=m.latency_ms,
                    error_rate=max(0.0, m.error_rate * 0.5),
                    saturation=min(m.saturation, cap),
                    availability=m.availability,
                )

        elif name == "ModifyConfig":
            service = params.get("service", "")
            if service and service in self.world_state.services:
                m = self.world_state.services[service]
                # Config change has minor positive effect
                from sentinel.models import ServiceMetrics
                self.world_state.services[service] = ServiceMetrics(
                    cpu=m.cpu,
                    memory=m.memory,
                    latency_ms=m.latency_ms,
                    error_rate=max(0.0, m.error_rate * 0.8),
                    saturation=max(0.05, m.saturation * 0.9),
                    availability=m.availability,
                )

        # ── Deployment actions ───────────────────────────────────────────
        elif name == "CanaryDeploy":
            service = params.get("service", "")
            traffic_percent = params.get("traffic_percent", 0.1)
            if service and service in self.world_state.services and incident_state is not None:
                from sentinel.models import TimelineEntry
                incident_state.timeline.append(
                    TimelineEntry(
                        step=self.step_count,
                        event_type="canary_deploy",
                        description=f"Canary deploy on {service} at {traffic_percent*100:.0f}% traffic",
                        agent=action.agent,
                    )
                )
                # Safe: canary has minimal impact, slight improvement
                m = self.world_state.services[service]
                from sentinel.models import ServiceMetrics
                improvement = traffic_percent * 0.3
                self.world_state.services[service] = ServiceMetrics(
                    cpu=m.cpu,
                    memory=m.memory,
                    latency_ms=max(10.0, m.latency_ms * (1.0 - improvement)),
                    error_rate=max(0.0, m.error_rate * (1.0 - improvement)),
                    saturation=m.saturation,
                    availability=m.availability,
                )

        elif name == "FullDeploy":
            service = params.get("service", "")
            if service and service in self.world_state.services and incident_state is not None:
                from sentinel.models import TimelineEntry
                incident_state.timeline.append(
                    TimelineEntry(
                        step=self.step_count,
                        event_type="full_deploy",
                        description=f"Full deploy on {service}",
                        agent=action.agent,
                    )
                )
                # Full deploy: if root cause is bad_deployment, fixes it
                if incident_state.failure_type.value == "bad_deployment" and service == incident_state.root_cause_service:
                    self._cascade_engine.propagate_recovery(
                        world_state=self.world_state,
                        resolved_service=service,
                    )
                    incident_state.current_blast_radius = self._cascade_engine.get_blast_radius()

        elif name == "Rollback":
            service = params.get("service", "")
            if service and service in self.world_state.services:
                from sentinel.world_state import _baseline_metrics
                self.world_state.services[service] = _baseline_metrics()
                if incident_state is not None:
                    incident_state.current_blast_radius.discard(service)

        # ── Meta actions ─────────────────────────────────────────────────
        elif name == "CloseIncident":
            if incident_state is not None:
                from sentinel.models import TimelineEntry
                incident_state.timeline.append(
                    TimelineEntry(
                        step=self.step_count,
                        event_type="close_incident",
                        description=params.get("resolution_summary", "Incident closed"),
                        agent=action.agent,
                    )
                )

        elif name == "EscalateToHuman":
            if incident_state is not None:
                from sentinel.models import TimelineEntry
                incident_state.timeline.append(
                    TimelineEntry(
                        step=self.step_count,
                        event_type="escalate",
                        description=f"Escalated: {params.get('reason', 'unknown')}",
                        agent=action.agent,
                    )
                )

        elif name == "GenerateNewScenario":
            if incident_state is not None:
                from sentinel.models import TimelineEntry
                incident_state.timeline.append(
                    TimelineEntry(
                        step=self.step_count,
                        event_type="generate_scenario",
                        description=f"Oracle generating scenario: {params.get('difficulty', 'medium')}",
                        agent=action.agent,
                    )
                )

        # Record attempted remediations
        if incident_state is not None and action.category == "remediation":
            incident_state.attempted_remediations.append(action)

    def _generate_agent_messages(self, action: Action) -> None:
        """Generate inter-agent messages based on the action taken.

        This implements the multi-agent coordination protocol:
        - HOLMES FormHypothesis → sends hypothesis to FORGE
        - FORGE remediation → sends result to ORACLE
        - ARGUS investigative → sends findings to HOLMES
        - ORACLE meta → broadcasts status to all
        """
        incident_state = self._incident_state
        if incident_state is None:
            return

        step = self.step_count

        if action.name == "FormHypothesis":
            # HOLMES → FORGE: ready for remediation
            self.message_bus.send(AgentMessage(
                sender="holmes",
                receiver="forge",
                message_type="hypothesis_confirmed",
                payload={
                    "service": action.params.get("service", ""),
                    "failure_type": action.params.get("failure_type", ""),
                    "confidence": action.params.get("confidence", 0.5),
                },
                step=step,
            ))

        elif action.category == "remediation":
            # FORGE → ORACLE: remediation result
            br_size = len(incident_state.current_blast_radius)
            self.message_bus.send(AgentMessage(
                sender="forge",
                receiver="oracle",
                message_type="remediation_result",
                payload={
                    "action": action.name,
                    "service": action.params.get("service", ""),
                    "blast_radius": br_size,
                },
                step=step,
            ))

        elif action.agent == "argus" and action.category == "investigative":
            # ARGUS → HOLMES: monitoring findings
            self.message_bus.send(AgentMessage(
                sender="argus",
                receiver="holmes",
                message_type="monitoring_update",
                payload={
                    "service": action.params.get("service", ""),
                    "action": action.name,
                },
                step=step,
            ))

        elif action.agent == "oracle":
            # ORACLE → broadcast
            self.message_bus.broadcast(
                sender="oracle",
                message_type="oracle_decision",
                payload={"action": action.name, "params": action.params},
                step=step,
            )

    def _propagate_secondary_failures(self) -> None:
        """Time-evolving cascade: unresolved failures spread to adjacent services.

        Called every 5 steps. For each currently-affected service, there's a
        20% chance each of its CDG neighbors gets mildly degraded (severity 0.3).
        This creates urgency — the agent can't investigate forever.
        """
        import random

        incident_state = self._incident_state
        if incident_state is None:
            return

        current_br = set(incident_state.current_blast_radius)
        if not current_br:
            return

        new_affected: dict[str, float] = {}
        for service in current_br:
            for neighbor in self.world_state.cdg.successors(service):
                if neighbor not in current_br and neighbor not in new_affected:
                    if random.random() < 0.2:  # 20% chance per neighbor
                        new_affected[neighbor] = 0.3

        # Apply secondary degradation
        for service, severity in new_affected.items():
            self.world_state.apply_degradation(service, severity)
            self._cascade_engine._affected_services[service] = severity
            incident_state.current_blast_radius.add(service)

    def _get_best_identification(self) -> tuple[str, str]:
        """Return the strongest current diagnosis for reward bookkeeping."""
        incident_state = self._incident_state
        if incident_state is None:
            return "", ""
        if incident_state.active_hypotheses:
            best = max(
                incident_state.active_hypotheses,
                key=lambda hypothesis: hypothesis.confidence,
            )
            return best.service, best.failure_type.value
        return "", ""
