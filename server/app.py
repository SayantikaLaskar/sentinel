"""FastAPI entrypoint for the OpenEnv validator and Hugging Face Space runtime."""
from __future__ import annotations

try:
    from openenv.core.env_server import create_app
except ModuleNotFoundError:
    from fastapi import FastAPI

from models import SentinelAction, SentinelObservation
from server.sentinel_environment import SentinelEnvironment

if "create_app" in globals():
    app = create_app(
        SentinelEnvironment,
        SentinelAction,
        SentinelObservation,
        env_name="sentinel",
        max_concurrent_envs=4,
    )
else:
    app = FastAPI(title="SENTINEL OpenEnv Compatibility Server")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "mode": "compatibility"}
