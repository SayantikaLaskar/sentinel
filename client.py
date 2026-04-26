"""OpenEnv client for connecting to a deployed SENTINEL Space."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
except ModuleNotFoundError:
    ActionT = TypeVar("ActionT")
    ObservationT = TypeVar("ObservationT")
    StateT = TypeVar("StateT")

    @dataclass
    class StepResult(Generic[ObservationT]):
        observation: ObservationT
        reward: float | None
        done: bool
        info: dict

    class EnvClient(Generic[ActionT, ObservationT, StateT]):
        """Local fallback when openenv-core is unavailable."""

        def __init__(self, *_args, **_kwargs) -> None:
            raise ModuleNotFoundError("openenv-core is required to use SentinelEnvClient.")

from models import SentinelAction, SentinelObservation, SentinelState


class SentinelEnvClient(EnvClient[SentinelAction, SentinelObservation, SentinelState]):
    """Typed OpenEnv client for SENTINEL."""

    def _step_payload(self, action: SentinelAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[SentinelObservation]:
        observation = SentinelObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            info=payload.get("info", {}),
        )

    def _parse_state(self, payload: dict) -> SentinelState:
        return SentinelState(**payload)
