"""Action parser for SENTINEL LLM agent.

The model is instructed to emit a single JSON object and nothing else.
This module extracts the first valid action-shaped object it can find,
repairs minor formatting issues when possible, and otherwise falls back
to a role-safe default action.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_VALID_AGENTS: frozenset[str] = frozenset(
    ["holmes", "forge", "argus", "hermes", "oracle"]
)

_VALID_CATEGORIES: frozenset[str] = frozenset(
    ["investigative", "remediation", "deployment", "meta"]
)

_INVESTIGATIVE_ACTIONS: frozenset[str] = frozenset([
    "QueryLogs", "QueryMetrics", "QueryTrace", "FormHypothesis",
])

_REMEDIATION_ACTIONS: frozenset[str] = frozenset([
    "RestartService", "ScaleService", "RollbackDeployment", "DrainTraffic",
    "ModifyRateLimit", "ModifyConfig",
])

_DEPLOYMENT_ACTIONS: frozenset[str] = frozenset([
    "CanaryDeploy", "FullDeploy", "Rollback",
])

_META_ACTIONS: frozenset[str] = frozenset([
    "CloseIncident", "EscalateToHuman", "GenerateNewScenario",
])

_ALL_ACTIONS: frozenset[str] = (
    _INVESTIGATIVE_ACTIONS | _REMEDIATION_ACTIONS | _DEPLOYMENT_ACTIONS | _META_ACTIONS
)

_CATEGORY_ACTIONS: dict[str, frozenset[str]] = {
    "investigative": _INVESTIGATIVE_ACTIONS,
    "remediation": _REMEDIATION_ACTIONS,
    "deployment": _DEPLOYMENT_ACTIONS,
    "meta": _META_ACTIONS,
}

_AGENT_CATEGORY: dict[str, str] = {
    "holmes": "investigative",
    "argus": "investigative",
    "forge": "remediation",
    "hermes": "deployment",
    "oracle": "meta",
}

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def parse_llm_action(
    llm_output: str,
    fallback_agent: str = "holmes",
) -> dict[str, Any]:
    action, _ = parse_llm_action_result(llm_output, fallback_agent=fallback_agent)
    return action


def parse_llm_action_result(
    llm_output: str,
    fallback_agent: str = "holmes",
) -> tuple[dict[str, Any], bool]:
    """Return (action_dict, parsed_ok)."""
    cleaned = _THINK_RE.sub("", llm_output).strip()

    action = _extract_from_code_block(cleaned)
    if action is None:
        action = _extract_raw_json(cleaned)
    if action is None:
        action = _extract_by_keyword(cleaned)
    if action is None:
        action = _extract_by_brace_walk(cleaned)
    if action is None:
        action = _extract_by_repair(cleaned)

    if action is None:
        logger.warning(
            "parse_llm_action: could not extract JSON from output (len=%d). Falling back to safe action.",
            len(llm_output),
        )
        return _safe_fallback(fallback_agent), False

    return _validate_and_repair(action, fallback_agent), True


def _extract_from_code_block(text: str) -> dict[str, Any] | None:
    match = _CODE_BLOCK_RE.search(text)
    if not match:
        return None
    return _try_parse(match.group(1).strip())


def _extract_raw_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start == -1:
        return None
    return _extract_balanced_object(text, start)


def _extract_by_keyword(text: str) -> dict[str, Any] | None:
    idx = text.find('"agent"')
    if idx == -1:
        idx = text.find("'agent'")
    if idx == -1:
        return None
    start = text.rfind("{", 0, idx)
    if start == -1:
        return None
    return _extract_balanced_object(text, start)


def _extract_by_brace_walk(text: str) -> dict[str, Any] | None:
    for i, ch in enumerate(text):
        if ch == "{":
            candidate = _extract_balanced_object(text, i)
            if candidate and "agent" in candidate and "name" in candidate:
                return candidate
    return None


def _extract_by_repair(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start == -1:
        return None
    fragment = text[start:].rstrip()
    fragment = fragment.rstrip("` \n\r\t")
    for extra in ("}", "}}", "}}}"):
        candidate = _try_parse(fragment + extra)
        if candidate and "agent" in candidate and "name" in candidate:
            return candidate
    return None


def _extract_balanced_object(text: str, start: int) -> dict[str, Any] | None:
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _try_parse(text[start : i + 1])
    return None


def _try_parse(raw: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        return None
    return None


def _validate_and_repair(action: dict[str, Any], fallback_agent: str) -> dict[str, Any]:
    agent = action.get("agent", fallback_agent)
    if agent not in _VALID_AGENTS:
        agent = fallback_agent

    name = action.get("name", "")
    if name not in _ALL_ACTIONS:
        lowered = str(name).lower()
        match = next((candidate for candidate in _ALL_ACTIONS if candidate.lower() == lowered), None)
        name = match or "QueryLogs"

    category = action.get("category", "")
    if category not in _VALID_CATEGORIES:
        category = _infer_category(name)

    if name not in _CATEGORY_ACTIONS.get(category, frozenset()):
        category = _infer_category(name)

    params = action.get("params", {})
    if not isinstance(params, dict):
        params = {}

    return {
        "agent": agent,
        "category": category,
        "name": name,
        "params": params,
    }


def _infer_category(name: str) -> str:
    if name in _INVESTIGATIVE_ACTIONS:
        return "investigative"
    if name in _REMEDIATION_ACTIONS:
        return "remediation"
    if name in _DEPLOYMENT_ACTIONS:
        return "deployment"
    return "meta"


def _safe_fallback(agent: str) -> dict[str, Any]:
    if agent == "forge":
        return {
            "agent": "forge",
            "category": "remediation",
            "name": "ScaleService",
            "params": {"service": "api-gateway", "replicas": 2},
        }
    if agent == "hermes":
        return {
            "agent": "hermes",
            "category": "deployment",
            "name": "Rollback",
            "params": {"service": "api-gateway"},
        }
    if agent == "oracle":
        return {
            "agent": "oracle",
            "category": "meta",
            "name": "EscalateToHuman",
            "params": {"reason": "LLM parse failure"},
        }
    return {
        "agent": "holmes",
        "category": "investigative",
        "name": "QueryLogs",
        "params": {"service": "api-gateway", "time_range": [0, 300]},
    }
