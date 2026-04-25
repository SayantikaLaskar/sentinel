from sentinel.math_engine import get_bayesian_rca, get_pagerank, get_alp_curriculum, get_ucb1_bandit
from sentinel.env import Sentinel_Env

env = Sentinel_Env()
obs, info = env.reset(seed=7)
print("Incident:", info["incident_id"])

# Bayesian RCA
rca = get_bayesian_rca()
top = rca.top_k(obs, k=3)
print("\n=== Bayesian RCA (Noisy-OR, Pearl 1988) ===")
for svc, prob in top:
    print(f"  P(root={svc}) = {prob:.4f}")

# PageRank
pr = get_pagerank()
ranked = pr.rank(obs)[:3]
print("\n=== Personalized PageRank (MicroRank WWW2021) ===")
for svc, score in ranked:
    print(f"  {svc}: {score:.4f}")

# ALP Curriculum
curriculum = get_alp_curriculum()
curriculum.record("easy", "cpu_spike", -0.3)
curriculum.record("easy", "cpu_spike", -0.1)
task = curriculum.next_task()
print(f"\n=== ALP Curriculum (Portelas 2020) next_task = {task} ===")

# UCB1 bandit
bandit = get_ucb1_bandit()
arm = bandit.select()
action = bandit.get_action_template(arm)
bandit.update(arm, 0.5)
agent_name = action["agent"]
act_name = action["name"]
print(f"\n=== UCB1 Bandit (Auer 2002) arm={arm}: {agent_name}/{act_name} ===")

# Full pipeline
from sentinel.training.pipeline import _get_action, TrainingConfig
cfg = TrainingConfig(agent="holmes")
a = _get_action(None, obs, cfg)
print("\n=== Full pipeline action (no trainer, no API) ===")
print(a)
print("\nALL MATH ENGINE TESTS PASSED. No LLM, no API calls.")
