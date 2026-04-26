"""OpenEnv wire models for the SENTINEL environment."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ModuleNotFoundError:
    class Action(BaseModel):
        """Local fallback when openenv-core is unavailable."""

    class Observation(BaseModel):
        """Local fallback when openenv-core is unavailable."""

        reward: float | None = None
        done: bool = False

    class State(BaseModel):
        """Local fallback when openenv-core is unavailable."""

        episode_id: str = ""
        step_count: int = 0


class SentinelAction(Action):
    """A single agent action emitted by the policy."""

    agent: str = Field(..., description="Acting agent id.")
    category: str = Field(..., description="Action category for role validation.")
    name: str = Field(..., description="Concrete action name.")
    params: dict[str, Any] = Field(default_factory=dict, description="Action parameters.")


class SentinelObservation(Observation):
    """Structured observation returned after reset/step."""

    metrics_snapshot: dict[str, Any] = Field(default_factory=dict)
    causal_graph_snapshot: list[float] = Field(default_factory=list)
    active_alerts: list[Any] = Field(default_factory=list)
    recent_logs: list[Any] = Field(default_factory=list)
    active_traces: list[Any] = Field(default_factory=list)
    incident_context: dict[str, Any] = Field(default_factory=dict)
    sla_state: dict[str, Any] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)


class SentinelState(State):
    """Episode metadata exposed through the OpenEnv state API."""

    incident_id: str | None = None
    terminated: bool = False
    truncated: bool = False
    identified_root_cause: str | None = None
    identified_failure_type: str | None = None
