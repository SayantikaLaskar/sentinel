"""Unified LLM client for SENTINEL.

Provides two clients driven by environment variables:

  HFInferenceClient  — calls meta-llama/Meta-Llama-3-8B-Instruct via the
                       HuggingFace Inference API to generate structured
                       agent actions from observations.

  OpenAIClient       — calls gpt-4o-mini to let the ORACLE agent synthesise
                       new IncidentTemplate parameters targeting capability gaps.

Both clients degrade gracefully to None when the relevant API key is absent,
allowing heuristic fallbacks to remain active.

Usage
-----
from sentinel.llm_client import get_hf_client, get_openai_client

hf  = get_hf_client()   # None if HF_TOKEN not set
oai = get_openai_client()  # None if OPENAI_API_KEY not set
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

# Load .env file if present (no-op when python-dotenv is not installed)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (overridable via env vars)
# ---------------------------------------------------------------------------

_HF_MODEL = os.environ.get(
    "HF_INFERENCE_MODEL",
    "meta-llama/Meta-Llama-3-8B-Instruct",
)
_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
_HEURISTIC_MODE = os.environ.get("SENTINEL_HEURISTIC_MODE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# HuggingFace Inference client
# ---------------------------------------------------------------------------

class HFInferenceClient:
    """Wraps huggingface_hub.InferenceClient for structured action generation."""

    def __init__(self, token: str) -> None:
        from huggingface_hub import InferenceClient  # type: ignore
        self._client = InferenceClient(model=_HF_MODEL, token=token)
        self._model = _HF_MODEL
        logger.info("HFInferenceClient initialised with model=%s", _HF_MODEL)

    def generate_action(self, observation: dict, agent_role: str) -> dict | None:
        """Ask Llama-3 to pick the next action for *agent_role* given *observation*.

        Returns a dict compatible with env.step() or None on failure.
        """
        prompt = _build_action_prompt(observation, agent_role)
        try:
            response = self._client.text_generation(
                prompt=prompt,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
            )
            return _parse_action_response(response, agent_role)
        except Exception as exc:
            logger.warning("HF inference failed (%s); heuristic will be used.", exc)
            return None


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

class OpenAIClient:
    """Wraps openai.OpenAI for Oracle scenario generation."""

    def __init__(self, api_key: str) -> None:
        import openai  # type: ignore
        self._client = openai.OpenAI(api_key=api_key)
        self._model = _OPENAI_MODEL
        logger.info("OpenAIClient initialised with model=%s", _OPENAI_MODEL)

    def generate_incident_params(
        self,
        capability_gap: str,
        difficulty: str,
        all_services: list[str],
        failure_types: list[str],
    ) -> dict | None:
        """Ask GPT-4o-mini to generate IncidentTemplate parameters.

        Returns a dict with keys: root_cause_service, failure_type,
        ground_truth_signals, red_herring_signals, cascade_risk,
        missing_log_ratio, expected_steps_to_resolve  — or None on failure.
        """
        prompt = _build_oracle_prompt(
            capability_gap, difficulty, all_services, failure_types
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert SRE designing training scenarios "
                            "for an autonomous incident-response AI system. "
                            "Always respond with valid JSON only — no markdown fences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as exc:
            logger.warning("OpenAI call failed (%s); default params will be used.", exc)
            return None


# ---------------------------------------------------------------------------
# Module-level singletons (lazy, created once)
# ---------------------------------------------------------------------------

_hf_client: HFInferenceClient | None = None
_oai_client: OpenAIClient | None = None
_clients_initialised = False


def _init_clients() -> None:
    global _hf_client, _oai_client, _clients_initialised
    if _clients_initialised:
        return
    _clients_initialised = True

    if _HEURISTIC_MODE:
        logger.info("SENTINEL_HEURISTIC_MODE=true — LLM clients disabled.")
        return

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            _hf_client = HFInferenceClient(token=hf_token)
        except Exception as exc:
            logger.warning("Could not create HFInferenceClient: %s", exc)

    oai_key = os.environ.get("OPENAI_API_KEY")
    if oai_key:
        try:
            _oai_client = OpenAIClient(api_key=oai_key)
        except Exception as exc:
            logger.warning("Could not create OpenAIClient: %s", exc)


def get_hf_client() -> HFInferenceClient | None:
    """Return the module-level HFInferenceClient, or None if unavailable."""
    _init_clients()
    return _hf_client


def get_openai_client() -> OpenAIClient | None:
    """Return the module-level OpenAIClient, or None if unavailable."""
    _init_clients()
    return _oai_client


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_action_prompt(observation: dict, agent_role: str) -> str:
    """Build a Llama-3 prompt for action selection."""
    # Summarise key observation fields
    incident_ctx_raw = observation.get("incident_context", "{}")
    if isinstance(incident_ctx_raw, str):
        try:
            incident_ctx = json.loads(incident_ctx_raw)
        except json.JSONDecodeError:
            incident_ctx = {}
    else:
        incident_ctx = incident_ctx_raw

    blast_radius = incident_ctx.get("current_blast_radius", [])
    hypotheses = incident_ctx.get("active_hypotheses", [])
    alerts_raw = observation.get("active_alerts", "[]")
    if isinstance(alerts_raw, str):
        try:
            alerts = json.loads(alerts_raw)
        except json.JSONDecodeError:
            alerts = []
    else:
        alerts = alerts_raw

    alert_services = list({
        a.get("service", "") if isinstance(a, dict) else getattr(a, "service", "")
        for a in alerts
    } - {""})

    role_instructions = {
        "holmes": (
            "You are HOLMES, the root-cause detective. "
            "Your allowed actions are: QueryLogs, QueryMetrics, QueryTrace, FormHypothesis. "
            "category must be 'investigative'."
        ),
        "forge": (
            "You are FORGE, the remediation executor. "
            "Your allowed actions are: RestartService, ScaleService, ModifyConfig, "
            "RollbackDeployment, DrainTraffic, ModifyRateLimit. "
            "category must be 'remediation'."
        ),
        "argus": (
            "You are ARGUS, the metric monitor. "
            "Your allowed actions are: QueryLogs, QueryMetrics, FormHypothesis. "
            "category must be 'investigative' or 'meta'."
        ),
        "hermes": (
            "You are HERMES, the deployment controller. "
            "Your allowed actions are: CanaryDeploy, FullDeploy, Rollback. "
            "category must be 'deployment'."
        ),
        "oracle": (
            "You are ORACLE, the self-improvement coordinator. "
            "Your allowed actions are: GenerateNewScenario, EscalateToHuman, CloseIncident. "
            "category must be 'meta'."
        ),
    }

    role_desc = role_instructions.get(agent_role, role_instructions["holmes"])

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an autonomous SRE agent in a multi-agent incident response system.
{role_desc}

Respond ONLY with a JSON object with exactly these keys:
  "agent": "{agent_role}"
  "category": <allowed category>
  "name": <action name>
  "params": <dict of parameters>

No explanation. No markdown. Pure JSON only.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Current incident state:
- Blast radius services: {blast_radius}
- Active alerts on: {alert_services}
- Active hypotheses: {hypotheses}

What is your next action?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def _parse_action_response(response: str, agent_role: str) -> dict | None:
    """Extract and validate a JSON action dict from the LLM response."""
    # Strip any accidental markdown fences
    text = response.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{"):
                text = part
                break

    # Find the first JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        action = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None

    # Validate required keys
    required = {"agent", "category", "name"}
    if not required.issubset(action.keys()):
        return None

    # Enforce correct agent field
    action["agent"] = agent_role
    if "params" not in action:
        action["params"] = {}

    return action


def _build_oracle_prompt(
    capability_gap: str,
    difficulty: str,
    all_services: list[str],
    failure_types: list[str],
) -> str:
    """Build a GPT-4o-mini prompt for Oracle scenario generation."""
    return f"""Design a new cloud incident scenario for training an autonomous SRE AI.

Requirements:
- difficulty: {difficulty}
- target capability gap: {capability_gap} (make the scenario hard in this dimension)
- available services: {all_services}
- available failure_types: {failure_types}

Return a JSON object with EXACTLY these keys:
{{
  "root_cause_service": "<one service from the list>",
  "failure_type": "<one failure type from the list>",
  "ground_truth_signals": ["<2-4 signal strings that reveal the root cause>"],
  "red_herring_signals": ["<1-3 misleading signal strings>"],
  "cascade_risk": "<low|medium|high>",
  "missing_log_ratio": <float 0.0-0.5>,
  "expected_steps_to_resolve": [<min_int>, <max_int>]
}}

For difficulty={difficulty}:
- easy: clear signals, no red herrings, low cascade_risk, missing_log_ratio < 0.1
- medium: some red herrings, medium cascade_risk, missing_log_ratio 0.1-0.3
- hard: many red herrings, high cascade_risk, missing_log_ratio 0.3-0.5, black-box root cause
"""
