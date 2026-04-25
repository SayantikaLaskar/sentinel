"""Mathematical intelligence layer for SENTINEL agents.

Implements research-backed algorithms replacing all LLM/API calls:

1. HOLMES — Bayesian Root Cause Analysis (Noisy-OR + Belief Propagation)
   Reference: Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems.
   Applied in microservices via MicroRank (WWW 2021) and CauseRank (AIOps).

2. FORGE — Personalized PageRank on dependency graph for remediation ranking
   Reference: MicroRank (WWW 2021): "Root Cause Localization of Cloud Native
   Microservice Systems with Personalized PageRank".
   PageRank math: Brin & Page (1998), modified for anomaly-biased random walks.

3. ORACLE — Absolute Learning Progress (ALP) Curriculum
   Reference: Portelas et al. (2020). "Teacher algorithms for curriculum
   learning of Deep RL in continuously parameterized environments." CoRL 2020.
   ALP = |R_t(c) - R_{t-Δt}(c)| — sample tasks maximising learning progress.

4. Training action selection — UCB1 bandit per action category
   Reference: Auer, Cesa-Bianchi & Fischer (2002). "Finite-time Analysis of
   the Multiarmed Bandit Problem." Machine Learning, 47, 235-256.
   UCB1 index: x_bar_i + sqrt(2 * ln(t) / n_i)
"""
from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Bayesian Noisy-OR Root Cause Analyser (Holmes)
# ---------------------------------------------------------------------------
# Based on Pearl (1988) Noisy-OR gate model for multi-cause diagnosis.
# Each alert a_j is modelled as:
#   P(a_j = 1 | parents) = 1 - prod_{i: parent active} (1 - q_ij)
# where q_ij is the "leak" probability that service i causes alert j.
# We run belief propagation to get posterior P(root_cause = s | alerts).
# ---------------------------------------------------------------------------

# Prior probability any service is the root cause (uniform)
_ROOT_CAUSE_PRIOR = 0.05   # 5% per service (30 services → sums to 1.5 > 1 intentionally; normalised)

# Noisy-OR leak parameter: P(alert fires | service i is failing)
_NOISY_OR_LEAK = 0.7

# Metric anomaly thresholds relative to baseline (used as evidence)
_METRIC_ANOMALY_THRESHOLDS = {
    "cpu":        0.7,    # > 70% CPU is anomalous
    "memory":     0.75,
    "error_rate": 0.05,
    "latency_ms": 200.0,
    "saturation": 0.8,
}


class BayesianRCA:
    """Noisy-OR Bayesian root cause analyser.

    Given a set of active alerts and metric snapshots, computes
    P(root_cause = s) for every service using Noisy-OR belief propagation.

    Mathematical basis
    ------------------
    For each candidate root cause service s and alert a:
        likelihood(a | s) = q_leak       if s is upstream of a
                          = q_baseline   otherwise

    posterior(s | alerts) ∝ prior(s) × ∏_a P(a | s)

    Upstream relationship is inferred from the adjacency matrix in the
    causal_graph_snapshot observation field.
    """

    def __init__(
        self,
        all_services: list[str],
        leak: float = _NOISY_OR_LEAK,
        prior: float = _ROOT_CAUSE_PRIOR,
    ) -> None:
        self.all_services = all_services
        self.n = len(all_services)
        self._svc_idx = {s: i for i, s in enumerate(all_services)}
        self.leak = leak
        self.prior = prior

    def infer(
        self,
        observation: dict,
    ) -> dict[str, float]:
        """Return posterior probability dict {service: prob} (normalised).

        Parameters
        ----------
        observation:
            Gymnasium observation dict as returned by Sentinel_Env.step().
        """
        # --- Parse adjacency matrix ---
        adj_raw = observation.get("causal_graph_snapshot")
        if adj_raw is not None and hasattr(adj_raw, "__len__") and len(adj_raw) == self.n * self.n:
            adj: list[list[float]] = [
                [float(adj_raw[i * self.n + j]) for j in range(self.n)]
                for i in range(self.n)
            ]
        else:
            adj = [[0.0] * self.n for _ in range(self.n)]

        # --- Parse active alerts ---
        alerts_raw = observation.get("active_alerts", "[]")
        if isinstance(alerts_raw, str):
            try:
                alerts = json.loads(alerts_raw)
            except json.JSONDecodeError:
                alerts = []
        else:
            alerts = list(alerts_raw)

        alerted_services: set[str] = set()
        for a in alerts:
            svc = a.get("service", "") if isinstance(a, dict) else getattr(a, "service", "")
            if svc:
                alerted_services.add(svc)

        # --- Parse metrics snapshot for additional evidence ---
        metrics_raw = observation.get("metrics_snapshot", "{}")
        if isinstance(metrics_raw, str):
            try:
                metrics_snap = json.loads(metrics_raw)
            except json.JSONDecodeError:
                metrics_snap = {}
        else:
            metrics_snap = metrics_raw or {}

        anomalous_services: set[str] = set()
        for svc, m in metrics_snap.items():
            if m is None:
                continue
            m_dict = m if isinstance(m, dict) else (m.__dict__ if hasattr(m, "__dict__") else {})
            if _is_metric_anomalous(m_dict):
                anomalous_services.add(svc)

        evidence_services = alerted_services | anomalous_services

        # --- Noisy-OR belief propagation ---
        posteriors: dict[str, float] = {}

        for cand_svc in self.all_services:
            cand_idx = self._svc_idx.get(cand_svc, -1)
            if cand_idx < 0:
                posteriors[cand_svc] = self.prior
                continue

            # Upstream services of cand_svc (services that cand_svc depends on)
            # In our adjacency: adj[i][j] > 0 means i → j (i causes j)
            # cand is upstream of alert_svc if adj[cand_idx][alert_svc_idx] > 0
            log_likelihood = 0.0

            for obs_svc in evidence_services:
                obs_idx = self._svc_idx.get(obs_svc, -1)
                if obs_idx < 0:
                    continue

                edge_weight = adj[cand_idx][obs_idx] if cand_idx < self.n and obs_idx < self.n else 0.0
                # Noisy-OR: P(obs_svc alerts | cand_svc is root) = leak * edge_weight
                # (if no edge: baseline leak of 0.1 from confounders)
                p_alert_given_root = max(self.leak * edge_weight, 0.1)
                p_alert_given_not_root = 0.05   # false alarm baseline
                # Log-likelihood ratio
                if p_alert_given_root > 0 and p_alert_given_not_root > 0:
                    log_likelihood += math.log(p_alert_given_root / p_alert_given_not_root)

            # Also add direct anomaly evidence: if cand_svc itself is anomalous
            if cand_svc in evidence_services:
                log_likelihood += math.log(self.leak / 0.05)

            posteriors[cand_svc] = self.prior * math.exp(log_likelihood)

        # Normalise
        total = sum(posteriors.values()) or 1.0
        return {s: v / total for s, v in posteriors.items()}

    def top_k(self, observation: dict, k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k most likely root cause services with probabilities."""
        posteriors = self.infer(observation)
        return sorted(posteriors.items(), key=lambda x: x[1], reverse=True)[:k]


def _is_metric_anomalous(m: dict) -> bool:
    """Return True if any metric field exceeds its anomaly threshold."""
    for field, threshold in _METRIC_ANOMALY_THRESHOLDS.items():
        val = m.get(field)
        if val is None:
            continue
        if float(val) > threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# 2. Personalized PageRank Remediation Ranker (Forge)
# ---------------------------------------------------------------------------
# Based on MicroRank (WWW 2021) and CauseRank (AIOps).
# Biases the random walk towards services already flagged as anomalous.
# PageRank iteration: r_{t+1} = α * A^T r_t + (1-α) * v
# where v is the personalisation vector (anomaly-weighted).
# ---------------------------------------------------------------------------

_PAGERANK_DAMPING = 0.85    # standard damping factor (Brin & Page 1998)
_PAGERANK_ITERATIONS = 50


class PersonalizedPageRank:
    """Personalized PageRank for remediation target ranking.

    Ranks services by their likelihood of being the best remediation target,
    using the dependency graph and anomaly evidence as a personalisation bias.

    Reference
    ---------
    MicroRank (WWW 2021): personalised PageRank on call-graph for RCL.
    Brin & Page (1998): original PageRank formulation.

    Iteration formula
    -----------------
        r_{t+1}(i) = α * Σ_j (A_{j→i} / out_degree(j)) * r_t(j)
                   + (1 - α) * v_i

    where v_i = anomaly_score(i) / Σ anomaly_score (personalisation vector).
    """

    def __init__(
        self,
        all_services: list[str],
        damping: float = _PAGERANK_DAMPING,
        iterations: int = _PAGERANK_ITERATIONS,
    ) -> None:
        self.all_services = all_services
        self.n = len(all_services)
        self._svc_idx = {s: i for i, s in enumerate(all_services)}
        self.damping = damping
        self.iterations = iterations

    def rank(
        self,
        observation: dict,
        bayesian_posteriors: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """Return services ranked by PPR score (best remediation target first).

        Parameters
        ----------
        bayesian_posteriors:
            If provided, uses Bayesian RCA scores as personalisation vector.
            Otherwise falls back to alert-count-based anomaly scores.
        """
        # --- Build adjacency matrix ---
        adj_raw = observation.get("causal_graph_snapshot")
        if adj_raw is not None and hasattr(adj_raw, "__len__") and len(adj_raw) == self.n * self.n:
            A = [
                [float(adj_raw[i * self.n + j]) for j in range(self.n)]
                for i in range(self.n)
            ]
        else:
            A = [[0.0] * self.n for _ in range(self.n)]

        # --- Build personalisation vector v ---
        if bayesian_posteriors:
            v = [bayesian_posteriors.get(s, 0.0) for s in self.all_services]
        else:
            # Fall back to alert-count scores
            alerts_raw = observation.get("active_alerts", "[]")
            if isinstance(alerts_raw, str):
                try:
                    alerts = json.loads(alerts_raw)
                except json.JSONDecodeError:
                    alerts = []
            else:
                alerts = list(alerts_raw)
            counts: dict[str, float] = defaultdict(float)
            for a in alerts:
                svc = a.get("service", "") if isinstance(a, dict) else getattr(a, "service", "")
                counts[svc] += 1.0
            v = [counts.get(s, 0.0) for s in self.all_services]

        v_sum = sum(v) or 1.0
        v = [x / v_sum for x in v]

        # --- Compute out-degree for normalisation ---
        out_degree = [sum(A[i]) or 1.0 for i in range(self.n)]

        # --- Initialise rank vector (uniform) ---
        r = [1.0 / self.n] * self.n

        # --- Power iteration ---
        for _ in range(self.iterations):
            new_r = [(1.0 - self.damping) * v[i] for i in range(self.n)]
            for j in range(self.n):
                if out_degree[j] > 0:
                    contrib = self.damping * r[j] / out_degree[j]
                    for i in range(self.n):
                        new_r[i] += contrib * A[j][i]
            r = new_r

        return sorted(
            zip(self.all_services, r),
            key=lambda x: x[1],
            reverse=True,
        )


# ---------------------------------------------------------------------------
# 3. ALP Curriculum Scheduler (Oracle)
# ---------------------------------------------------------------------------
# Based on Portelas et al. (2020) CoRL — Absolute Learning Progress.
# ALP_t(c) = |R_t(c) - R_{t-Δt}(c)|
# Sample tasks c that maximise ALP (zone of proximal development).
# ---------------------------------------------------------------------------

class ALPCurriculum:
    """Absolute Learning Progress curriculum scheduler.

    Tracks performance history per (difficulty, failure_type) task cell
    and selects the next task that maximises |ΔR| — the absolute change
    in reward, signalling maximum current learning opportunity.

    Reference
    ---------
    Portelas, R., Colas, C., Weng, L., Hofmann, K., & Oudeyer, P.-Y. (2020).
    Teacher algorithms for curriculum learning of Deep RL in continuously
    parameterized environments. CoRL 2020.
    """

    DIFFICULTIES = ("easy", "medium", "hard")
    FAILURE_TYPES = (
        "memory_leak", "connection_pool_exhaustion", "cpu_spike",
        "bad_deployment", "cache_miss_storm", "network_partition",
    )

    def __init__(self, window: int = 5) -> None:
        """
        Parameters
        ----------
        window:
            Number of recent rewards to track per task cell for ALP computation.
        """
        self.window = window
        # reward_history[(difficulty, failure_type)] = list of recent total rewards
        self.reward_history: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._episode_count: int = 0

    def record(
        self,
        difficulty: str,
        failure_type: str,
        total_reward: float,
    ) -> None:
        """Record the outcome of a completed episode."""
        key = (difficulty, failure_type)
        hist = self.reward_history[key]
        hist.append(total_reward)
        if len(hist) > self.window:
            hist.pop(0)
        self._episode_count += 1

    def alp(self, difficulty: str, failure_type: str) -> float:
        """Compute ALP for a task cell.

        ALP = |mean(last half) - mean(first half)|  (window-smoothed version).
        Returns 1.0 for unexplored cells (exploration bonus).
        """
        key = (difficulty, failure_type)
        hist = self.reward_history[key]
        if len(hist) < 2:
            return 1.0   # unexplored → maximum priority
        mid = len(hist) // 2
        first_half = hist[:mid]
        second_half = hist[mid:]
        return abs(
            sum(second_half) / len(second_half) -
            sum(first_half) / len(first_half)
        )

    def next_task(self) -> tuple[str, str]:
        """Return (difficulty, failure_type) with highest ALP.

        Ties are broken by epsilon-greedy exploration (ε=0.1) to ensure
        all cells are eventually visited.
        """
        epsilon = 0.1
        if random.random() < epsilon:
            # Explore: pick a random unexplored or least-explored cell
            return (
                random.choice(self.DIFFICULTIES),
                random.choice(self.FAILURE_TYPES),
            )

        best_key: tuple[str, str] = (self.DIFFICULTIES[0], self.FAILURE_TYPES[0])
        best_alp = -1.0
        for diff in self.DIFFICULTIES:
            for ft in self.FAILURE_TYPES:
                a = self.alp(diff, ft)
                if a > best_alp:
                    best_alp = a
                    best_key = (diff, ft)
        return best_key

    def summary(self) -> dict[str, dict[str, float]]:
        """Return ALP scores for all cells (for logging/plotting)."""
        out: dict[str, dict[str, float]] = {}
        for diff in self.DIFFICULTIES:
            out[diff] = {}
            for ft in self.FAILURE_TYPES:
                out[diff][ft] = round(self.alp(diff, ft), 4)
        return out


# ---------------------------------------------------------------------------
# 4. UCB1 Action Bandit (Training Loop)
# ---------------------------------------------------------------------------
# Auer, Cesa-Bianchi & Fischer (2002). Finite-time Analysis of the
# Multiarmed Bandit Problem. Machine Learning, 47, 235-256.
# Index: x_bar_i + sqrt(2 * ln(t) / n_i)
# Selects which high-level action category to explore this episode.
# ---------------------------------------------------------------------------

# Action templates per category — observation-driven parameters filled later
_ACTION_TEMPLATES: dict[str, list[dict]] = {
    "investigative": [
        {"agent": "holmes", "category": "investigative", "name": "QueryLogs",     "params": {}},
        {"agent": "holmes", "category": "investigative", "name": "QueryMetrics",  "params": {}},
        {"agent": "holmes", "category": "investigative", "name": "QueryTrace",    "params": {}},
        {"agent": "holmes", "category": "investigative", "name": "FormHypothesis","params": {}},
        {"agent": "argus",  "category": "investigative", "name": "QueryMetrics",  "params": {}},
    ],
    "remediation": [
        {"agent": "forge", "category": "remediation", "name": "RestartService",      "params": {}},
        {"agent": "forge", "category": "remediation", "name": "ScaleService",        "params": {}},
        {"agent": "forge", "category": "remediation", "name": "RollbackDeployment",  "params": {}},
        {"agent": "forge", "category": "remediation", "name": "DrainTraffic",        "params": {}},
        {"agent": "forge", "category": "remediation", "name": "ModifyRateLimit",     "params": {}},
    ],
    "meta": [
        {"agent": "oracle", "category": "meta", "name": "CloseIncident",         "params": {}},
        {"agent": "oracle", "category": "meta", "name": "EscalateToHuman",       "params": {}},
        {"agent": "oracle", "category": "meta", "name": "GenerateNewScenario",   "params": {}},
    ],
}


class UCB1ActionBandit:
    """UCB1 bandit for multi-step action selection during training.

    Each "arm" is a (category, action_name) pair.
    The bandit tracks empirical rewards and exploration bonuses per arm,
    selecting arms that balance exploitation (high mean reward) and
    exploration (rarely tried arms).

    UCB1 index (Auer et al. 2002)
    ─────────────────────────────
        I_i(t) = x̄_i + √(2 · ln(t) / n_i)

    where x̄_i = mean reward of arm i,
          n_i  = number of pulls,
          t    = total pulls across all arms.
    """

    def __init__(self) -> None:
        # Flatten all arms
        self._arms: list[dict] = []
        for templates in _ACTION_TEMPLATES.values():
            for t in templates:
                self._arms.append(dict(t))

        self._n_arms = len(self._arms)
        self._counts = [0] * self._n_arms      # n_i
        self._rewards = [0.0] * self._n_arms   # cumulative reward per arm
        self._t = 0                             # total pulls

    def select(self) -> int:
        """Return index of the arm with highest UCB1 index.

        During initialisation (any arm with n_i=0), cycles through arms
        to ensure each is tried once before UCB1 kicks in.
        """
        # Initialisation phase: try each arm once
        for i, cnt in enumerate(self._counts):
            if cnt == 0:
                return i

        self._t += 1
        best_idx = 0
        best_ucb = -float("inf")
        for i in range(self._n_arms):
            mean_reward = self._rewards[i] / self._counts[i]
            exploration = math.sqrt(2.0 * math.log(self._t) / self._counts[i])
            ucb = mean_reward + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i
        return best_idx

    def update(self, arm_idx: int, reward: float) -> None:
        """Update empirical statistics for the chosen arm."""
        self._counts[arm_idx] += 1
        self._rewards[arm_idx] += reward
        if self._t == 0:
            self._t = 1

    def get_action_template(self, arm_idx: int) -> dict:
        """Return the action template for the given arm index."""
        return dict(self._arms[arm_idx])

    def arm_stats(self) -> list[dict]:
        """Return statistics for all arms (for logging/plotting)."""
        stats = []
        for i, arm in enumerate(self._arms):
            mean = self._rewards[i] / self._counts[i] if self._counts[i] > 0 else 0.0
            stats.append({
                "arm": f"{arm['agent']}/{arm['name']}",
                "pulls": self._counts[i],
                "mean_reward": round(mean, 4),
            })
        return sorted(stats, key=lambda x: x["mean_reward"], reverse=True)


# ---------------------------------------------------------------------------
# Module-level singletons (lazy)
# ---------------------------------------------------------------------------

_bayesian_rca: BayesianRCA | None = None
_pagerank: PersonalizedPageRank | None = None
_alp_curriculum: ALPCurriculum | None = None
_ucb1_bandit: UCB1ActionBandit | None = None


def get_bayesian_rca(all_services: list[str] | None = None) -> BayesianRCA:
    global _bayesian_rca
    if _bayesian_rca is None:
        from sentinel.world_state import ALL_SERVICES
        _bayesian_rca = BayesianRCA(all_services or list(ALL_SERVICES))
    return _bayesian_rca


def get_pagerank(all_services: list[str] | None = None) -> PersonalizedPageRank:
    global _pagerank
    if _pagerank is None:
        from sentinel.world_state import ALL_SERVICES
        _pagerank = PersonalizedPageRank(all_services or list(ALL_SERVICES))
    return _pagerank


def get_alp_curriculum() -> ALPCurriculum:
    global _alp_curriculum
    if _alp_curriculum is None:
        _alp_curriculum = ALPCurriculum()
    return _alp_curriculum


def get_ucb1_bandit() -> UCB1ActionBandit:
    global _ucb1_bandit
    if _ucb1_bandit is None:
        _ucb1_bandit = UCB1ActionBandit()
    return _ucb1_bandit
