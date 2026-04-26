"""SENTINEL validation dashboard for Hugging Face Spaces."""
from __future__ import annotations

import json
from typing import Any

from models import SentinelAction
from sentinel.config import load_config
from server.sentinel_environment import SentinelEnvironment

try:
    import gradio as gr
    _GRADIO_AVAILABLE = True
except ImportError:
    gr = None  # type: ignore[assignment]
    _GRADIO_AVAILABLE = False

_adapter: SentinelEnvironment | None = None
_last_observation: dict[str, Any] = {}
_last_step_result: dict[str, Any] = {}
_action_log: list[str] = []
_INCIDENT_IDS = ["E1", "E2", "E3", "M1", "M2", "M3", "M4", "H1", "H2", "H3"]

_PRESET_ACTIONS: dict[str, dict[str, Any]] = {
    "QueryLogs": {
        "agent": "holmes",
        "category": "investigative",
        "name": "QueryLogs",
        "params": {"service": "cart-service", "time_range": [0, 60]},
    },
    "QueryMetrics": {
        "agent": "argus",
        "category": "investigative",
        "name": "QueryMetrics",
        "params": {"service": "order-service", "metric_name": "cpu", "time_range": [0, 60]},
    },
    "RestartService": {
        "agent": "forge",
        "category": "remediation",
        "name": "RestartService",
        "params": {"service": "cart-service"},
    },
    "EscalateToHuman": {
        "agent": "oracle",
        "category": "meta",
        "name": "EscalateToHuman",
        "params": {"reason": "Validator smoke test escalation"},
    },
}


def _ensure_adapter() -> SentinelEnvironment:
    global _adapter
    if _adapter is None:
        _adapter = SentinelEnvironment()
    return _adapter


def _sync_adapter_state(
    adapter: SentinelEnvironment,
    observation: Any,
    *,
    episode_id: str | None = None,
) -> None:
    adapter._state.episode_id = episode_id or adapter._env._episode_id
    adapter._state.step_count = adapter._env.step_count
    adapter._state.incident_id = observation.info.get("incident_id", adapter._state.incident_id)
    adapter._state.terminated = bool(observation.done)
    adapter._state.truncated = False


def _json_blob(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _append_log(message: str) -> None:
    _action_log.append(message)
    del _action_log[:-20]


def _health_summary() -> str:
    if _adapter is None:
        return (
            '<div style="display:flex;gap:12px;flex-wrap:wrap;">'
            '<div style="background:#1e293b;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;text-align:center;">'
            '<div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Status</div>'
            '<div style="color:#f59e0b;font-size:20px;font-weight:700;margin-top:4px;">Idle</div></div>'
            '<div style="background:#1e293b;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;text-align:center;">'
            '<div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Steps</div>'
            '<div style="color:#e2e8f0;font-size:20px;font-weight:700;margin-top:4px;">—</div></div>'
            '<div style="background:#1e293b;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;text-align:center;">'
            '<div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Incident</div>'
            '<div style="color:#e2e8f0;font-size:20px;font-weight:700;margin-top:4px;">—</div></div>'
            '<div style="background:#1e293b;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;text-align:center;">'
            '<div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Reward</div>'
            '<div style="color:#e2e8f0;font-size:20px;font-weight:700;margin-top:4px;">—</div></div>'
            '</div>'
        )

    state = _adapter.state.model_dump()
    step_count = state['step_count']
    incident = state.get('incident_id') or '—'
    terminated = state.get('terminated', False)

    status_color = "#ef4444" if terminated else "#22c55e"
    status_label = "Done" if terminated else "Active"

    last_reward = "—"
    if _last_step_result and "reward" in _last_step_result:
        r = _last_step_result["reward"]
        if isinstance(r, (int, float)):
            last_reward = f"{r:+.3f}"

    return (
        '<div style="display:flex;gap:12px;flex-wrap:wrap;">'
        f'<div style="background:#1e293b;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;text-align:center;">'
        f'<div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Status</div>'
        f'<div style="color:{status_color};font-size:20px;font-weight:700;margin-top:4px;">{status_label}</div></div>'
        f'<div style="background:#1e293b;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;text-align:center;">'
        f'<div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Steps</div>'
        f'<div style="color:#e2e8f0;font-size:20px;font-weight:700;margin-top:4px;">{step_count}</div></div>'
        f'<div style="background:#1e293b;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;text-align:center;">'
        f'<div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Incident</div>'
        f'<div style="color:#38bdf8;font-size:20px;font-weight:700;margin-top:4px;">{incident}</div></div>'
        f'<div style="background:#1e293b;border-radius:10px;padding:14px 20px;flex:1;min-width:140px;text-align:center;">'
        f'<div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Last Reward</div>'
        f'<div style="color:#e2e8f0;font-size:20px;font-weight:700;margin-top:4px;">{last_reward}</div></div>'
        '</div>'
    )


def _service_health_html() -> str:
    if _adapter is None:
        return (
            '<div style="background:#0f172a;border-radius:12px;padding:24px;text-align:center;color:#64748b;">'
            '<p style="font-size:14px;">🔌 Environment not initialized. Click <b>Reset Episode</b> to start.</p>'
            '</div>'
        )

    env = _adapter._env
    incident_state = env.world_state.incident_state
    blast_radius = incident_state.current_blast_radius if incident_state is not None else set()
    root_cause = incident_state.root_cause_service if incident_state is not None else None

    healthy = 0
    degraded = 0
    cells: list[str] = []
    for service, metrics in env.world_state.services.items():
        is_blast = service in blast_radius
        is_root = service == root_cause

        if not metrics.availability:
            bg = "#7f1d1d"
            border_color = "#ef4444"
            icon = "🔴"
            degraded += 1
        elif is_blast:
            bg = "#78350f"
            border_color = "#f59e0b"
            icon = "🟡"
            degraded += 1
        else:
            bg = "#064e3b"
            border_color = "#10b981"
            icon = "🟢"
            healthy += 1

        root_badge = ' <span style="background:#dc2626;color:#fff;font-size:9px;padding:1px 5px;border-radius:4px;vertical-align:middle;">ROOT</span>' if is_root else ""
        cpu_color = "#ef4444" if metrics.cpu > 0.8 else "#f59e0b" if metrics.cpu > 0.5 else "#10b981"
        err_color = "#ef4444" if metrics.error_rate > 0.05 else "#f59e0b" if metrics.error_rate > 0.01 else "#10b981"

        cells.append(
            f'<div style="background:{bg};border:2px solid {border_color};border-radius:8px;'
            f'padding:10px 12px;margin:3px;display:inline-block;width:170px;vertical-align:top;'
            f'color:#fff;font:12px/1.5 ui-monospace,monospace;">'
            f'<div style="font-weight:700;font-size:13px;margin-bottom:4px;">{icon} {service}{root_badge}</div>'
            f'<div>CPU <span style="color:{cpu_color};font-weight:600;">{metrics.cpu * 100:.0f}%</span></div>'
            f'<div>Err <span style="color:{err_color};font-weight:600;">{metrics.error_rate * 100:.1f}%</span></div>'
            f'<div>Lat <span style="font-weight:600;">{metrics.latency_ms:.0f}ms</span></div>'
            '</div>'
        )

    summary = (
        f'<div style="display:flex;gap:16px;margin-bottom:12px;font-size:13px;color:#cbd5e1;">'
        f'<span>🟢 Healthy: <b>{healthy}</b></span>'
        f'<span>🔴 Degraded: <b>{degraded}</b></span>'
        f'<span>📊 Total: <b>{healthy + degraded}</b></span>'
        f'</div>'
    )

    return (
        f'<div style="background:#0f172a;padding:16px;border-radius:12px;">'
        f'{summary}'
        f'<div style="display:flex;flex-wrap:wrap;gap:0;">'
        + "".join(cells)
        + '</div></div>'
    )


def _render_snapshot() -> str:
    if _adapter is None:
        return "(no render available yet)"
    rendered = _adapter._env.render()
    return rendered or "(render returned no output)"


def _current_state_json() -> str:
    if _adapter is None:
        return _json_blob({"initialized": False})
    return _json_blob(_adapter.state.model_dump())


def _current_log_text() -> str:
    if not _action_log:
        return "(no actions yet)"
    return "\n".join(reversed(_action_log))


def _snapshot(status: str) -> tuple[str, str, str, str, str, str, str]:
    return (
        status,
        _health_summary(),
        _service_health_html(),
        _current_state_json(),
        _json_blob(_last_observation),
        _json_blob(_last_step_result),
        _current_log_text(),
    )


def _reset_env(seed: int, incident_id: str) -> tuple[str, str, str, str, str, str, str]:
    global _last_observation, _last_step_result

    adapter = _ensure_adapter()
    try:
        obs, info = adapter._env.reset(seed=seed, options={"incident_id": incident_id})
        observation = adapter._to_observation(obs=obs, reward=0.0, done=False, info=info)
        _sync_adapter_state(adapter, observation)
    except Exception as exc:
        return _snapshot(f"Reset failed: `{exc}`")

    _last_observation = observation.model_dump()
    _last_step_result = {
        "event": "reset",
        "seed": seed,
        "incident_id": adapter.state.incident_id,
        "done": observation.done,
    }
    _append_log(f"reset(seed={seed}, incident_id={adapter.state.incident_id})")
    return _snapshot(
        f"> ✅ **Reset succeeded** — Episode `{adapter.state.episode_id[:8]}…` ready. Incident **{adapter.state.incident_id}** loaded."
    )


def _run_action(action_name: str) -> tuple[str, str, str, str, str, str, str]:
    global _last_observation, _last_step_result

    adapter = _ensure_adapter()
    payload = _PRESET_ACTIONS[action_name]
    try:
        action = SentinelAction(**payload)
        observation = adapter.step(action)
    except Exception as exc:
        _append_log(f"{payload['agent']}::{payload['name']} -> ERROR: {exc}")
        return _snapshot(f"Action `{payload['name']}` failed: `{exc}`")

    _last_observation = observation.model_dump()
    _last_step_result = {
        "action": payload,
        "reward": observation.reward,
        "done": observation.done,
        "info": observation.info,
    }
    _append_log(
        f"{payload['agent']}::{payload['name']} -> reward={observation.reward:.3f}, done={observation.done}"
    )
    reward_icon = "🟢" if observation.reward > 0 else "🔴" if observation.reward < -0.2 else "🟡"
    return _snapshot(
        f"> {reward_icon} **{payload['agent'].title()}** → `{payload['name']}` — Reward: **{observation.reward:+.3f}** | Done: **{observation.done}**"
    )


def _run_custom_action(action_json: str) -> tuple[str, str, str, str, str, str, str]:
    global _last_observation, _last_step_result

    adapter = _ensure_adapter()
    try:
        payload = json.loads(action_json)
        action = SentinelAction(**payload)
    except Exception as exc:
        return _snapshot(f"Custom action parse failed: `{exc}`")

    try:
        observation = adapter.step(action)
    except Exception as exc:
        _append_log(f"custom_action -> ERROR: {exc}")
        return _snapshot(f"Custom action execution failed: `{exc}`")

    _last_observation = observation.model_dump()
    _last_step_result = {
        "action": payload,
        "reward": observation.reward,
        "done": observation.done,
        "info": observation.info,
    }
    _append_log(
        f"{payload['agent']}::{payload['name']} -> reward={observation.reward:.3f}, done={observation.done}"
    )
    reward_icon = "🟢" if observation.reward > 0 else "🔴" if observation.reward < -0.2 else "🟡"
    return _snapshot(
        f"> {reward_icon} **Custom** → `{payload['name']}` — Reward: **{observation.reward:+.3f}** | Done: **{observation.done}**"
    )


def _run_smoke_test(seed: int, incident_id: str) -> tuple[str, str, str, str, str, str, str]:
    global _last_observation, _last_step_result

    adapter = SentinelEnvironment()
    try:
        obs, info = adapter._env.reset(seed=seed, options={"incident_id": incident_id})
        observation = adapter._to_observation(obs=obs, reward=0.0, done=False, info=info)
        _sync_adapter_state(adapter, observation)
        query_logs = adapter.step(SentinelAction(**_PRESET_ACTIONS["QueryLogs"]))
        query_metrics = adapter.step(SentinelAction(**_PRESET_ACTIONS["QueryMetrics"]))
    except Exception as exc:
        _last_step_result = {"smoke_test": "fail", "error": str(exc)}
        _append_log(f"smoke_test(seed={seed}, incident={incident_id}) -> ERROR: {exc}")
        return _snapshot(f"Smoke test failed: `{exc}`")

    required_keys = {
        "metrics_snapshot",
        "causal_graph_snapshot",
        "active_alerts",
        "recent_logs",
        "active_traces",
        "incident_context",
        "sla_state",
        "reward",
        "done",
        "info",
    }
    missing = sorted(required_keys.difference(query_metrics.model_dump().keys()))

    ok = not missing and adapter.state.step_count >= 2
    _adapter_state = adapter.state.model_dump()
    _last_observation = query_metrics.model_dump()
    _last_step_result = {
        "smoke_test": "pass" if ok else "fail",
        "reset_incident_id": observation.info.get("incident_id"),
        "query_logs_reward": query_logs.reward,
        "query_metrics_reward": query_metrics.reward,
        "state": _adapter_state,
        "missing_keys": missing,
    }
    _append_log(
        f"smoke_test(seed={seed}, incident={incident_id}) -> {'PASS' if ok else 'FAIL'}"
    )

    status = (
        "> ✅ **Smoke test PASSED** — `reset`, `step`, `state`, and observation schema all verified."
        if ok
        else f"> ❌ **Smoke test FAILED** — Missing keys: {missing}"
    )
    return _snapshot(status)


def build_dashboard() -> Any:
    if not _GRADIO_AVAILABLE:
        return None

    cfg = load_config()
    default_action = _json_blob(cfg.training.placeholder_action)

    css = """
    .gradio-container { max-width: 1200px !important; }
    .status-banner { padding: 10px 16px; border-radius: 8px; font-size: 14px; }
    """

    with gr.Blocks(
        title="SENTINEL — Incident Response Environment",
    ) as dashboard:

        # ── Header ────────────────────────────────────────────
        gr.HTML(
            '<div style="text-align:center;padding:20px 0 8px;">'
            '<h1 style="margin:0;font-size:32px;">🛡️ SENTINEL</h1>'
            '<p style="color:#64748b;margin:6px 0 0;font-size:15px;">'
            'Multi-Agent Incident Response Environment &mdash; '
            '<a href="/health" target="_blank">/health</a> · '
            '<a href="/docs" target="_blank">/docs</a> · '
            '<a href="https://github.com/sayantikalaskar/sentinel" target="_blank">GitHub</a>'
            '</p></div>'
        )

        # ── Status bar ────────────────────────────────────────
        status = gr.Markdown(
            value='> ⏳ **Ready** — Click "Reset Episode" to initialize the environment.',
            elem_classes=["status-banner"],
        )

        # ── KPI cards ─────────────────────────────────────────
        runtime = gr.HTML(value=_health_summary())

        # ── Controls ──────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1, min_width=120):
                seed = gr.Number(value=cfg.demo.seed, precision=0, label="Seed")
            with gr.Column(scale=1, min_width=120):
                incident_id = gr.Dropdown(
                    choices=_INCIDENT_IDS,
                    value=_INCIDENT_IDS[0],
                    label="Incident",
                )
            with gr.Column(scale=1, min_width=160):
                reset_btn = gr.Button("▶  Reset Episode", variant="primary", size="lg")
            with gr.Column(scale=1, min_width=160):
                smoke_btn = gr.Button("🧪  Smoke Test", variant="secondary", size="lg")

        # ── Service Health Grid ───────────────────────────────
        with gr.Accordion("🏥 Service Health Grid (30 services)", open=True):
            health_grid = gr.HTML(value=_service_health_html())

        # ── Agent Actions ─────────────────────────────────────
        gr.Markdown("### 🎮 Agent Actions")
        gr.Markdown(
            "*Each button sends a preset action from a specific agent role. "
            "Hover for details.*",
        )
        with gr.Row():
            query_logs_btn = gr.Button("🔍 Holmes: Query Logs", size="sm")
            query_metrics_btn = gr.Button("📊 Argus: Query Metrics", size="sm")
            restart_btn = gr.Button("🔧 Forge: Restart Service", size="sm")
            escalate_btn = gr.Button("🚨 Oracle: Escalate", size="sm")

        # ── Two-column: results + log ─────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("📋 Step Result"):
                        latest_step = gr.Code(
                            value=_json_blob(_last_step_result),
                            language="json",
                            label="Latest Step Result",
                        )
                    with gr.Tab("👁 Observation"):
                        observation_json = gr.Code(
                            value=_json_blob(_last_observation),
                            language="json",
                            label="Observation",
                        )
                    with gr.Tab("🔧 State"):
                        state_json = gr.Code(
                            value=_current_state_json(),
                            language="json",
                            label="Adapter State",
                        )
                    with gr.Tab("🎬 Render"):
                        render_output = gr.Textbox(
                            value=_render_snapshot(),
                            label="Environment Render",
                            lines=10,
                            interactive=False,
                        )
            with gr.Column(scale=2):
                action_log = gr.Textbox(
                    value=_current_log_text(),
                    label="📜 Action Log",
                    lines=14,
                    interactive=False,
                )

        # ── Custom Action ─────────────────────────────────────
        with gr.Accordion("⚡ Custom Action (Advanced)", open=False):
            gr.Markdown(
                "Send any valid JSON action. The `agent` field must match a valid role "
                "(`holmes`, `forge`, `argus`, `hermes`, `oracle`)."
            )
            custom_action = gr.Code(
                value=default_action,
                language="json",
                label="Action JSON",
            )
            custom_btn = gr.Button("Run Custom Action", variant="secondary")

        # ── How It Works ──────────────────────────────────────
        with gr.Accordion("ℹ️  How This Works", open=False):
            gr.Markdown(
                """
**SENTINEL** simulates a 30-service cloud platform experiencing an outage.

**Workflow:**
1. **Reset** — Injects a failure into the service graph. Cascading damage spreads.
2. **Investigate** — Holmes queries logs/metrics to find the root cause.
3. **Remediate** — Forge restarts/scales/rolls back affected services.
4. **Resolve** — Oracle escalates or closes the incident.

**Reward Signal:**
- R1 (35%): Did the agent identify the correct root cause?
- R2 (30%): How fast was the resolution (MTTR)?
- R3 (25%): How fully did services recover?
- R4 (10%): Was blast radius contained?

**Color Legend:**
🟢 Healthy — 🟡 In blast radius — 🔴 Down/Unavailable — `ROOT` = root cause service
"""
            )

        # ── Wire up events ────────────────────────────────────
        outputs = [status, runtime, health_grid, state_json, observation_json, latest_step, action_log]

        reset_btn.click(fn=_reset_env, inputs=[seed, incident_id], outputs=outputs)
        smoke_btn.click(fn=_run_smoke_test, inputs=[seed, incident_id], outputs=outputs)
        query_logs_btn.click(fn=lambda: _run_action("QueryLogs"), outputs=outputs)
        query_metrics_btn.click(fn=lambda: _run_action("QueryMetrics"), outputs=outputs)
        restart_btn.click(fn=lambda: _run_action("RestartService"), outputs=outputs)
        escalate_btn.click(fn=lambda: _run_action("EscalateToHuman"), outputs=outputs)
        custom_btn.click(fn=_run_custom_action, inputs=[custom_action], outputs=outputs)

        for trigger in (
            reset_btn,
            smoke_btn,
            query_logs_btn,
            query_metrics_btn,
            restart_btn,
            escalate_btn,
            custom_btn,
        ):
            trigger.click(fn=_render_snapshot, outputs=[render_output])

    return dashboard


demo = build_dashboard()

if __name__ == "__main__" and demo is not None:
    demo.launch(share=False)
