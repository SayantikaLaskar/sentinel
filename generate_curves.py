#!/usr/bin/env python3
"""Standalone plot generation — can be run anytime during or after training."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
AGENTS = ["holmes", "forge", "argus", "hermes", "oracle"]


def smooth(arr, window=5):
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def load_log(agent, tag="full"):
    log_file = RESULTS_DIR / f"{agent}_{tag}_log.jsonl"
    if not log_file.exists():
        print(f"  [SKIP] {log_file} not found")
        return None
    episodes, rewards, r1s, r2s, r3s, r4s, mttrs, losses = [], [], [], [], [], [], [], []
    with open(log_file) as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                episodes.append(rec["episode"])
                rewards.append(rec["total_reward"])
                r1s.append(rec["r1"])
                r2s.append(rec.get("r2", 0))
                r3s.append(rec.get("r3", 0))
                r4s.append(rec.get("r4", 0))
                mttrs.append(rec["mttr"])
                losses.append(rec.get("loss", 0))
            except (json.JSONDecodeError, KeyError):
                continue
    if not episodes:
        return None
    return {
        "episodes": np.array(episodes),
        "rewards": np.array(rewards),
        "r1": np.array(r1s),
        "r2": np.array(r2s),
        "r3": np.array(r3s),
        "r4": np.array(r4s),
        "mttrs": np.array(mttrs),
        "losses": np.array([l if l is not None else 0.0 for l in losses]),
    }


def plot_agent(agent, data):
    n_eps = len(data["episodes"])
    episodes = data["episodes"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"SENTINEL Training — {agent.upper()} Agent ({n_eps} episodes)",
        fontsize=14, fontweight="bold"
    )

    # --- Total reward ---
    ax = axes[0, 0]
    ax.plot(episodes, data["rewards"], alpha=0.3, color="blue", label="Raw")
    if n_eps >= 5:
        sm = smooth(data["rewards"])
        ax.plot(episodes[:len(sm)], sm, color="blue", linewidth=2, label="Smoothed (w=5)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Total Reward per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- R1 Root Cause ---
    ax = axes[0, 1]
    ax.plot(episodes, data["r1"], alpha=0.3, color="green", label="Raw")
    if n_eps >= 5:
        sm = smooth(data["r1"])
        ax.plot(episodes[:len(sm)], sm, color="green", linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("R1 (Root Cause)")
    ax.set_title("R1 Root Cause Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- MTTR ---
    ax = axes[1, 0]
    ax.plot(episodes, data["mttrs"], alpha=0.3, color="red", label="Raw")
    if n_eps >= 5:
        sm = smooth(data["mttrs"])
        ax.plot(episodes[:len(sm)], sm, color="red", linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("MTTR (steps)")
    ax.set_title("Mean Time To Resolve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Reward Breakdown ---
    ax = axes[1, 1]
    ax.plot(episodes, data["r1"], label="R1 Root Cause (0.35)", alpha=0.7)
    ax.plot(episodes, data["r2"], label="R2 MTTR (0.30)", alpha=0.7)
    ax.plot(episodes, data["r3"], label="R3 Recovery (0.25)", alpha=0.7)
    ax.plot(episodes, data["r4"], label="R4 Blast Radius (0.10)", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward Component")
    ax.set_title("Reward Breakdown")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / f"{agent}_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out}")

    # --- Loss curve (separate) ---
    valid_losses = np.array([l if l is not None else 0.0 for l in data["losses"]])
    if any(valid_losses != 0.0):
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(episodes, valid_losses, alpha=0.3, color="purple", label="Raw")
        if n_eps >= 5:
            sm = smooth(valid_losses)
            ax2.plot(episodes[:len(sm)], sm, color="purple", linewidth=2, label="Smoothed (w=5)")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.set_title(f"SENTINEL — {agent.upper()} Agent Training Loss ({n_eps} episodes)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        out2 = RESULTS_DIR / f"{agent}_loss_curve.png"
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  ✓ Saved {out2}")


def plot_comparison(agent_data):
    agents_with_data = {a: d for a, d in agent_data.items() if d is not None}
    if len(agents_with_data) < 1:
        print("  [SKIP] Not enough data for comparison")
        return

    colors = {"holmes": "#2196F3", "forge": "#FF9800", "argus": "#4CAF50", "hermes": "#9C27B0", "oracle": "#F44336"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("SENTINEL — All Agents Training Comparison", fontsize=13, fontweight="bold")

    for agent, data in agents_with_data.items():
        eps = data["episodes"]
        c = colors.get(agent, "gray")

        sm_r = smooth(data["rewards"], 7) if len(data["rewards"]) >= 7 else data["rewards"]
        axes[0].plot(eps[:len(sm_r)], sm_r, color=c, linewidth=2, label=f"{agent.capitalize()} ({len(eps)} eps)")

        sm_r1 = smooth(data["r1"], 7) if len(data["r1"]) >= 7 else data["r1"]
        axes[1].plot(eps[:len(sm_r1)], sm_r1, color=c, linewidth=2, label=f"{agent.capitalize()}")

        sm_m = smooth(data["mttrs"], 7) if len(data["mttrs"]) >= 7 else data["mttrs"]
        axes[2].plot(eps[:len(sm_m)], sm_m, color=c, linewidth=2, label=f"{agent.capitalize()}")

    for ax, title, ylabel in zip(axes, ["Total Reward", "R1 Root Cause", "MTTR"],
                                       ["Reward", "R1", "Steps"]):
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "comparison_all_agents.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out}")

    # Loss comparison
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for agent, data in agents_with_data.items():
        valid_losses = np.array([l if l is not None else 0.0 for l in data["losses"]])
        if any(valid_losses != 0.0):
            c = colors.get(agent, "gray")
            eps = data["episodes"]
            sm = smooth(valid_losses, 7) if len(valid_losses) >= 7 else valid_losses
            ax2.plot(eps[:len(sm)], sm, color=c, linewidth=2, label=f"{agent.capitalize()} ({len(eps)} eps)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("SENTINEL — Training Loss Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = RESULTS_DIR / "comparison_loss.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Saved {out2}")


if __name__ == "__main__":
    print("=" * 60)
    print("SENTINEL — Generating Training Curves")
    print("=" * 60)

    agent_data = {}
    for agent in AGENTS:
        print(f"\n[{agent.upper()}]")
        data = load_log(agent, "full")
        agent_data[agent] = data
        if data is not None:
            print(f"  Loaded {len(data['episodes'])} episodes")
            plot_agent(agent, data)
        else:
            print("  No data available")

    print(f"\n[COMPARISON]")
    plot_comparison(agent_data)

    print("\n" + "=" * 60)
    print("Done! All plots saved to:", RESULTS_DIR)
    print("=" * 60)
