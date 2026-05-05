"""
Diagnostic script for the cold-chain PPO routing agent.

Runs 100 episodes each for three policies:
  1. Random      — pure baseline
  2. Greedy      — nearest-neighbour heuristic (30-line rule-based policy)
  3. PPO         — trained agent

Outputs:
  - Reward decomposition table (transport / spoilage / stockout per policy)
  - Scale check  (are cost terms on the same magnitude?)
  - results/diagnosis_comparison.png
  - Verdict: is PPO adding value over a simple heuristic?
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.rl_routing.environment import ColdChainRoutingEnv
from stable_baselines3 import PPO

PLOTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models', 'plots')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rl_routing', 'ppo_cold_chain')
N_EPISODES = 100


# ── Greedy policy ─────────────────────────────────────────────────────────────

class GreedyPolicy:
    """
    Nearest-neighbour heuristic:
      1. Find the warehouse with the largest demand-inventory gap (most urgent).
      2. Among all routes that touch that warehouse, pick the one with the
         shortest distance and lowest weather severity.
    """
    ROUTES  = ColdChainRoutingEnv.ROUTES
    MAX_KM  = ColdChainRoutingEnv.MAX_DISTANCE

    def predict(self, obs, deterministic=True):
        inv_ratio    = obs[0:6]
        demand_ratio = obs[6:12]
        weather      = obs[12:18]

        urgency          = demand_ratio - inv_ratio          # higher = more desperate
        most_urgent_wh   = int(np.argmax(urgency))

        best_action, best_score = 0, -np.inf
        for i, route in enumerate(self.ROUTES):
            a, b = route['pair']
            if a == most_urgent_wh or b == most_urgent_wh:
                route_weather = float(max(weather[a], weather[b]))
                score = -(route['distance_km'] / self.MAX_KM) - route_weather * 0.5
                if score > best_score:
                    best_score, best_action = score, i

        return best_action, None


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episodes(policy, n=N_EPISODES, deterministic=True, seed_offset=0):
    env     = ColdChainRoutingEnv()
    results = []
    for ep in range(n):
        obs, _ = env.reset(seed=ep + seed_offset)
        done   = False
        ep_data = {'reward': 0., 'transport': 0., 'spoilage': 0., 'stockout': 0.,
                   'route_counts': np.zeros(15, dtype=int)}
        while not done:
            action, _ = policy.predict(obs, deterministic=deterministic)
            action     = int(action)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_data['reward']    += reward
            ep_data['transport'] += info['transport_cost']
            ep_data['spoilage']  += info['spoilage_cost']
            ep_data['stockout']  += info['stockout_cost']
            ep_data['route_counts'][action] += 1
        results.append(ep_data)
    env.close()
    return results


class RandomPolicy:
    def __init__(self):
        self._env = ColdChainRoutingEnv()
    def predict(self, obs, deterministic=False):
        return self._env.action_space.sample(), None


# ── Summary stats ─────────────────────────────────────────────────────────────

def summarise(results):
    keys = ['reward', 'transport', 'spoilage', 'stockout']
    return {k: {'mean': np.mean([r[k] for r in results]),
                'std':  np.std( [r[k] for r in results])} for k in keys}


def print_table(stats: dict):
    header = f"{'Policy':<10} {'Reward':>10} {'Transport':>12} {'Spoilage':>12} {'Stockout':>12}"
    print("\n" + header)
    print("─" * len(header))
    for name, s in stats.items():
        print(f"{name:<10} "
              f"{s['reward']['mean']:>10.2f} "
              f"{s['transport']['mean']:>12.3f} "
              f"{s['spoilage']['mean']:>12.3f} "
              f"{s['stockout']['mean']:>12.3f}")


def scale_check(stats: dict):
    ppo = stats['PPO']
    t = abs(ppo['transport']['mean'])
    s = abs(ppo['spoilage']['mean'])
    k = abs(ppo['stockout']['mean'])
    total = t + s + k
    print(f"\n── PPO cost breakdown (% of total penalty) ─────────────────")
    print(f"  Transport : {t/total*100:5.1f}%  (raw={t:.3f})")
    print(f"  Spoilage  : {s/total*100:5.1f}%  (raw={s:.3f})")
    print(f"  Stockout  : {k/total*100:5.1f}%  (raw={k:.3f})")
    if k / max(s, 1e-9) > 3:
        print("\n  ⚠  Stockout dominates (>3× spoilage). Agent under-weights cold-chain.")
        print("     Fix: raise C_STOCKOUT further OR lower C_STOCKOUT so spoilage is relatively louder.")
    else:
        print("\n  ✓  Cost terms are reasonably balanced.")


def verdict(stats: dict):
    r_r = stats['Random']['reward']['mean']
    g_r = stats['Greedy']['reward']['mean']
    p_r = stats['PPO']['reward']['mean']

    greedy_vs_random = (g_r - r_r) / abs(r_r) * 100
    ppo_vs_greedy    = (p_r - g_r) / abs(g_r) * 100
    ppo_vs_random    = (p_r - r_r) / abs(r_r) * 100

    print(f"\n── Verdict ──────────────────────────────────────────────────")
    print(f"  Greedy vs Random : {greedy_vs_random:+.1f}%")
    print(f"  PPO    vs Random : {ppo_vs_random:+.1f}%")
    print(f"  PPO    vs Greedy : {ppo_vs_greedy:+.1f}%")

    if ppo_vs_greedy > 3:
        print("\n  ✓  PPO meaningfully beats greedy — RL is adding real value.")
    elif ppo_vs_greedy > 0:
        print("\n  ~  PPO edges out greedy but margin is small.")
        print("     Consider: tighter reward scaling, more training, or curriculum.")
    else:
        print("\n  ✗  PPO does NOT beat greedy. Greedy heuristic is stronger.")
        print("     This is a useful finding — the environment may be too stochastic")
        print("     for PPO to learn a policy better than a simple rule.")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_comparison(stats: dict, all_results: dict):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    policies = list(stats.keys())
    colors   = {'Random': 'tomato', 'Greedy': 'goldenrod', 'PPO': 'steelblue'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Policy Comparison: Random vs Greedy vs PPO', fontsize=13, fontweight='bold')

    # ── Cost breakdown grouped bar ────────────────────────────────────────
    ax   = axes[0]
    cats = ['transport', 'spoilage', 'stockout']
    x    = np.arange(len(cats))
    w    = 0.25
    for i, pol in enumerate(policies):
        vals = [stats[pol][c]['mean'] for c in cats]
        ax.bar(x + (i - 1) * w, vals, w, label=pol, color=colors[pol], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(['Transport', 'Spoilage', 'Stockout'])
    ax.set_title('Mean Cost per Episode'); ax.set_ylabel('Cost'); ax.legend()

    # ── Reward distribution ────────────────────────────────────────────────
    ax = axes[1]
    for pol in policies:
        rewards = [r['reward'] for r in all_results[pol]]
        ax.hist(rewards, bins=25, alpha=0.55, color=colors[pol],
                label=f"{pol} μ={np.mean(rewards):.1f}")
    ax.set_title('Episode Reward Distribution')
    ax.set_xlabel('Total Reward'); ax.set_ylabel('Frequency'); ax.legend()

    # ── Route preference: PPO vs Greedy ──────────────────────────────────
    ax     = axes[2]
    env    = ColdChainRoutingEnv()
    labels = [f"{env.WAREHOUSE_NAMES[r['pair'][0]][:3]}↔"
              f"{env.WAREHOUSE_NAMES[r['pair'][1]][:3]}" for r in env.ROUTES]
    env.close()

    for pol, offset, col in [('Greedy', -0.2, 'goldenrod'), ('PPO', 0.2, 'steelblue')]:
        counts = np.zeros(15)
        for r in all_results[pol]:
            counts += r['route_counts']
        counts /= counts.sum()
        ax.bar(np.arange(15) + offset, counts, 0.38, label=pol, color=col, alpha=0.75)

    ax.set_xticks(np.arange(15)); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.axhline(1/15, color='grey', linestyle='--', linewidth=1, label='Uniform')
    ax.set_title('Route Preferences: Greedy vs PPO'); ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, 'diagnosis_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading PPO model…")
    ppo_model = PPO.load(MODEL_PATH)

    random_policy = RandomPolicy()
    greedy_policy = GreedyPolicy()

    print(f"Running {N_EPISODES} episodes per policy…")
    results = {
        'Random': run_episodes(random_policy, deterministic=False, seed_offset=0),
        'Greedy': run_episodes(greedy_policy, seed_offset=0),
        'PPO':    run_episodes(ppo_model,     seed_offset=0),
    }

    stats = {name: summarise(res) for name, res in results.items()}

    print_table(stats)
    scale_check(stats)
    verdict(stats)
    plot_comparison(stats, results)

    # Update summary.json with diagnosis results
    import json
    summary_path = os.path.join(os.path.dirname(__file__), 'summary.json')
    with open(summary_path) as f:
        summary = json.load(f)

    summary['phase_3_rl_routing']['diagnosis'] = {
        'n_episodes': N_EPISODES,
        'random_mean_reward':  round(stats['Random']['reward']['mean'], 3),
        'greedy_mean_reward':  round(stats['Greedy']['reward']['mean'], 3),
        'ppo_mean_reward':     round(stats['PPO']['reward']['mean'], 3),
        'ppo_vs_greedy_pct':   round((stats['PPO']['reward']['mean'] - stats['Greedy']['reward']['mean'])
                                     / abs(stats['Greedy']['reward']['mean']) * 100, 1),
        'cost_breakdown_ppo': {
            'transport': round(stats['PPO']['transport']['mean'], 3),
            'spoilage':  round(stats['PPO']['spoilage']['mean'],  3),
            'stockout':  round(stats['PPO']['stockout']['mean'],  3),
        }
    }
    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return super().default(o)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=_Encoder)
    print(f"  Updated → results/summary.json")


if __name__ == '__main__':
    main()
