"""Prompt builder for SENTINEL LLM agents."""
from __future__ import annotations

import json
from typing import Any

_ACTION_SCHEMA = """\
Return exactly one JSON object and nothing else.
Do not use markdown fences.
Do not explain your reasoning.
Do not add extra keys.

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
        "Investigate the most suspicious service and produce a valid investigative action."
    ),
    "forge": (
        "You are FORGE, the remediation engineer for NexaStack. "
        "Apply the safest valid remediation to reduce blast radius and recover service health."
    ),
    "argus": (
        "You are ARGUS, the monitoring specialist for NexaStack. "
        "Surface the highest-signal evidence with a valid investigative action."
    ),
    "hermes": (
        "You are HERMES, the deployment controller for NexaStack. "
        "Use valid deployment actions only when rollout state is relevant."
    ),
    "oracle": (
        "You are ORACLE, the escalation and closure controller for NexaStack. "
        "Close only when the incident is resolved, otherwise escalate or generate a scenario."
    ),
}


def build_prompt(
    obs: dict[str, Any],
    agent_role: str = "holmes",
    step_number: int = 0,
) -> str:
    system = _SYSTEM_PROMPTS.get(agent_role, _SYSTEM_PROMPTS["holmes"])
    user_block = _format_observation(obs, step_number)
    return (
        f"{system}\n\n"
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
    user_block = _format_observation(obs, step_number)
    return [
        {"role": "system", "content": f"{system}\n\n{_ACTION_SCHEMA}"},
        {"role": "user", "content": user_block},
        {"role": "assistant", "content": "{"},
    ]


def _format_observation(obs: dict[str, Any], step_number: int) -> str:
    lines: list[str] = [f"Step: {step_number}"]

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

    lines.append(
        'Example valid output: {"agent":"holmes","category":"investigative","name":"QueryLogs","params":{"service":"postgres-primary","time_range":[0,300]}}'
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
