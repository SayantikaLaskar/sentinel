"""OpenEnv adapter that exposes Sentinel_Env through the current OpenEnv API."""
from __future__ import annotations

import json
from uuid import uuid4

from typing import Generic, TypeVar

try:
    from openenv.core.env_server.interfaces import Environment
except ModuleNotFoundError:
    ActionT = TypeVar("ActionT")
    ObservationT = TypeVar("ObservationT")
    StateT = TypeVar("StateT")

    class Environment(Generic[ActionT, ObservationT, StateT]):
        """Local fallback when openenv-core is unavailable."""

from models import SentinelAction, SentinelObservation, SentinelState
from sentinel.env import Sentinel_Env


class SentinelEnvironment(Environment[SentinelAction, SentinelObservation, SentinelState]):
    """Thin OpenEnv wrapper around the existing Gym-style Sentinel_Env."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._env = Sentinel_Env(render_mode="json")
        self._state = SentinelState(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> SentinelObservation:
        obs, info = self._env.reset(seed=seed)
        self._state = SentinelState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            incident_id=info.get("incident_id"),
            terminated=False,
            truncated=False,
        )
        return self._to_observation(obs=obs, reward=0.0, done=False, info=info)

    def step(self, action: SentinelAction) -> SentinelObservation:
        obs, reward, terminated, truncated, info = self._env.step(action.model_dump())
        self._state.step_count = self._env.step_count
        self._state.terminated = terminated
        self._state.truncated = truncated
        self._state.identified_root_cause = info.get("identified_root_cause")
        self._state.identified_failure_type = info.get("identified_failure_type")
        return self._to_observation(
            obs=obs,
            reward=reward,
            done=terminated or truncated,
            info=info,
        )

    @property
    def state(self) -> SentinelState:
        return self._state

    def _to_observation(
        self,
        *,
        obs: dict,
        reward: float,
        done: bool,
        info: dict,
    ) -> SentinelObservation:
        return SentinelObservation(
            metrics_snapshot=self._decode(obs.get("metrics_snapshot")),
            causal_graph_snapshot=self._tolist(obs.get("causal_graph_snapshot")),
            active_alerts=self._decode(obs.get("active_alerts"), default=[]),
            recent_logs=self._decode(obs.get("recent_logs"), default=[]),
            active_traces=self._decode(obs.get("active_traces"), default=[]),
            incident_context=self._decode(obs.get("incident_context")),
            sla_state=self._decode(obs.get("sla_state")),
            reward=reward,
            done=done,
            info=info,
        )

    @staticmethod
    def _tolist(value: object) -> list[float]:
        if hasattr(value, "tolist"):
            return list(value.tolist())
        if isinstance(value, list):
            return value
        return []

    @staticmethod
    def _decode(value: object, default: object | None = None) -> object:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default if default is not None else {}
        return value if value is not None else (default if default is not None else {})
