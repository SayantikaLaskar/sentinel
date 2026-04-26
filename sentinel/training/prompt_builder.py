"""Prompt builder for SENTINEL LLM agents."""
from __future__ import annotations

import json
from typing import Any

_ACTION_SCHEMA = """\
Required schema:
{
  "agent": "<holmes|forge|argus|hermes|oracle>",
  "category": "<investigative|remediation|deployment|meta>",
  "name": "<valid action name>",
  "params": {}
}

Valid actions by role:
- holmes or argus:
  QueryLogs params={service, time_range:[start,end]}
  QueryMetrics params={service, metric_name, time_range:[start,end]}
  QueryTrace params={trace_id}
  FormHypothesis params={service, failure_type, confidence}
- forge:
  RestartService params={service}
  ScaleService params={service, replicas}
  RollbackDeployment params={service, version}
  DrainTraffic params={service}
  ModifyRateLimit params={service, limit_rps}
  ModifyConfig params={service, key, value}
- hermes:
  CanaryDeploy params={service, version, traffic_percent}
  FullDeploy params={service, version}
  Rollback params={service}
- oracle:
  CloseIncident params={resolution_summary}
  EscalateToHuman params={reason}
  GenerateNewScenario params={difficulty, target_gap}

Valid failure_type values:
cpu_spike, memory_leak, bad_deployment, connection_pool_exhaustion, cache_miss_storm, network_partition
"""

_SYSTEM_PROMPTS: dict[str, str] = {
    "holmes": (
        "You are HOLMES, the root-cause analyst for NexaStack. "
        "Your goal: identify which service is the ROOT CAUSE and what failure_type it has.\n"
        "STRATEGY: First, use QueryLogs and QueryMetrics on the most degraded services shown in alerts. "
        "Look at error_rate, latency_ms, and availability. The service with the highest error_rate "
        "and lowest availability is likely the root cause.\n"
        "THEN: Once you have a suspect, call FormHypothesis with {service, failure_type, confidence}. "
        "Valid failure_type values: cpu_spike, memory_leak, bad_deployment, "
        "connection_pool_exhaustion, cache_miss_storm, network_partition.\n"
        "FormHypothesis RESOLVES the incident. A correct diagnosis earns maximum reward. "
        "Do NOT just keep querying — form your hypothesis as soon as you have evidence."
    ),
    "forge": (
        "You are FORGE, the remediation engineer for NexaStack. "
        "Your goal: reduce the blast radius to ZERO by fixing affected services.\n"
        "STRATEGY: Look at the blast radius services and degraded services in the observation. "
        "Use RestartService on the most degraded services first. "
        "For bad_deployment failures, use RollbackDeployment. "
        "For high saturation, use ScaleService with replicas=3.\n"
        "Target services that appear in BOTH the blast radius AND alerts. "
        "The incident resolves when blast radius reaches 0. Act fast — fewer steps = higher reward."
    ),
    "argus": (
        "You are ARGUS, the monitoring specialist for NexaStack. "
        "Your goal: identify which service is the ROOT CAUSE and what failure_type it has.\n"
        "STRATEGY: Use QueryLogs and QueryMetrics on services with the highest error_rate in alerts. "
        "Cross-reference metrics to distinguish root cause from downstream symptoms.\n"
        "THEN: Once you have a suspect, call FormHypothesis with {service, failure_type, confidence}. "
        "Valid failure_type values: cpu_spike, memory_leak, bad_deployment, "
        "connection_pool_exhaustion, cache_miss_storm, network_partition.\n"
        "FormHypothesis RESOLVES the incident. Form your hypothesis as soon as you have evidence."
    ),
    "hermes": (
        "You are HERMES, the deployment controller for NexaStack. "
        "Your goal: fix deployment-related failures using CanaryDeploy, FullDeploy, or Rollback.\n"
        "STRATEGY: Look at degraded services in alerts. If the incident looks like a bad_deployment, "
        "use Rollback on the most degraded service first. For other failures, use CanaryDeploy "
        "to test a fix at low traffic, then FullDeploy to roll it out.\n"
        "The episode resolves when blast radius reaches 0 after your deployment action. "
        "Act decisively — fewer steps = higher reward."
    ),
    "oracle": (
        "You are ORACLE, the escalation and closure controller for NexaStack. "
        "Your goal: decide whether the incident should be CLOSED or ESCALATED to a human.\n"
        "STRATEGY: Check the blast radius count and SLA state. "
        "If blast_radius is 0 or very low and SLA is not breached, call CloseIncident. "
        "If the situation is worsening or blast radius is growing, call EscalateToHuman with a reason.\n"
        "CloseIncident earns reward when the incident is truly resolved. "
        "EscalateToHuman is the safe choice when you are unsure — it ends the episode cleanly."
    ),
}


_SYSTEM_SUFFIX = (
    "\n\nIMPORTANT: Output ONLY one JSON object. No explanation, no markdown, no text before or after. "
    "Your entire response must be a single JSON object under 60 tokens."
)


def build_prompt(
    obs: dict[str, Any],
    agent_role: str = "holmes",
    step_number: int = 0,
) -> str:
    system = _SYSTEM_PROMPTS.get(agent_role, _SYSTEM_PROMPTS["holmes"])
    user_block = _format_observation(obs, step_number, agent_role)
    return (
        f"{system}{_SYSTEM_SUFFIX}\n\n"
        f"{_ACTION_SCHEMA}\n\n"
        f"{user_block}\n\n"
        "Output JSON now:"
    )


def build_messages(
    obs: dict[str, Any],
    agent_role: str = "holmes",
    step_number: int = 0,
) -> list[dict[str, str]]:
    system = _SYSTEM_PROMPTS.get(agent_role, _SYSTEM_PROMPTS["holmes"])
    user_block = _format_observation(obs, step_number, agent_role)
    return [
        {"role": "system", "content": f"{system}{_SYSTEM_SUFFIX}\n\n{_ACTION_SCHEMA}"},
        {"role": "user", "content": user_block},
        {"role": "assistant", "content": "{"},
    ]


def _format_observation(obs: dict[str, Any], step_number: int, agent_role: str = "holmes") -> str:
    lines: list[str] = [f"Step: {step_number}"]

    # Phase hints to guide action selection
    if agent_role in ("holmes", "argus"):
        if step_number >= 5:
            lines.append(">>> You have investigated enough. Use FormHypothesis NOW. <<<")
        elif step_number >= 3:
            lines.append(">>> Consider forming a hypothesis soon based on the evidence below. <<<")
    elif agent_role == "forge":
        if step_number >= 5:
            lines.append(">>> Act NOW — remediate the most degraded services to reduce blast radius to 0. <<<")
    elif agent_role == "hermes":
        if step_number >= 5:
            lines.append(">>> Act NOW — use Rollback or FullDeploy on degraded services. <<<")
        elif step_number >= 3:
            lines.append(">>> Consider deploying a fix or rolling back the bad deployment. <<<")
    elif agent_role == "oracle":
        if step_number >= 5:
            lines.append(">>> Decide NOW — CloseIncident if blast radius is 0, otherwise EscalateToHuman. <<<")
        elif step_number >= 3:
            lines.append(">>> Assess the situation — is the incident resolving or worsening? <<<")

    inc = _parse_json_field(obs.get("incident_context", "{}"))
    if inc:
        blast = inc.get("current_blast_radius", [])
        lines.append(f"Blast radius count: {len(blast)}")
        if blast:
            lines.append(f"Blast radius services: {', '.join(blast[:6])}")
        hypotheses = inc.get("active_hypotheses", [])
        if hypotheses:
            lines.append("Current hypotheses:")
            for hypothesis in hypotheses[:3]:
                if isinstance(hypothesis, dict):
                    lines.append(
                        f"- {hypothesis.get('service', '?')} / {hypothesis.get('failure_type', '?')} / conf={hypothesis.get('confidence', 0)}"
                    )

    alerts = _parse_json_field(obs.get("active_alerts", "[]"))
    if alerts:
        lines.append("Top alerts:")
        for alert in alerts[:5]:
            if isinstance(alert, dict):
                lines.append(
                    f"- {alert.get('service', '?')} {alert.get('metric', alert.get('metric_name', '?'))}={alert.get('value', alert.get('current_value', '?'))}"
                )

    metrics = _parse_json_field(obs.get("metrics_snapshot", "{}"))
    degraded = _find_degraded(metrics)
    if degraded:
        lines.append("Degraded services:")
        for service, metric in degraded[:6]:
            lines.append(
                f"- {service}: err={metric.get('error_rate', 0):.3f} lat={metric.get('latency_ms', 0):.0f} avail={metric.get('availability', True)} sat={metric.get('saturation', 0):.2f}"
            )

    logs = _parse_json_field(obs.get("recent_logs", "[]"))
    if logs:
        lines.append("Recent logs:")
        for log in logs[:4]:
            if isinstance(log, dict):
                lines.append(f"- [{log.get('service', '?')}] {str(log.get('message', ''))[:100]}")

    sla = _parse_json_field(obs.get("sla_state", "{}"))
    if sla:
        lines.append(
            f"SLA breached={sla.get('breached', False)} current_mttr={sla.get('current_mttr', step_number)} blast_radius={sla.get('blast_radius', 0)}"
        )

    return "\n".join(lines)


def _parse_json_field(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {} if raw.strip().startswith("{") else []
    return {}


def _find_degraded(metrics: Any) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(metrics, dict):
        return []
    results: list[tuple[str, dict[str, Any]]] = []
    for service, metric in metrics.items():
        if not isinstance(metric, dict):
            continue
        err = float(metric.get("error_rate", 0) or 0)
        latency = float(metric.get("latency_ms", 0) or 0)
        available = bool(metric.get("availability", True))
        if err > 0.05 or latency > 300 or not available:
            results.append((service, metric))
    results.sort(key=lambda item: -(float(item[1].get("error_rate", 0) or 0)))
    return results
