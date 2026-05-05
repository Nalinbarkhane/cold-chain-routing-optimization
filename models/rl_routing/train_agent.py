import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from models.rl_routing.environment import ColdChainRoutingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

SAVE_DIR  = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
MODEL_OUT = os.path.join(SAVE_DIR, 'ppo_cold_chain')
LOG_DIR   = os.path.join(SAVE_DIR, 'logs')
TOTAL_TIMESTEPS = 2_000_000


# ── Reward logger callback ────────────────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """Collects mean episode reward at each rollout for the learning curve."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._ep_reward_buf   = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
        return True


# ── Policy evaluation helper ──────────────────────────────────────────────────

def run_episodes(policy, n_episodes=50, deterministic=True):
    """
    Roll out `policy` (PPO model or 'random') for n_episodes.
    Returns list of per-episode dicts with total reward and cost breakdown.
    """
    env = ColdChainRoutingEnv()
    results = []

    for _ in range(n_episodes):
        obs, _     = env.reset()
        done       = False
        ep_reward  = 0.0
        ep_costs   = {'transport': 0., 'spoilage': 0., 'stockout': 0.}
        route_counts = np.zeros(15, dtype=int)

        while not done:
            if policy == 'random':
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs, deterministic=deterministic)
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward             += reward
            ep_costs['transport'] += info['transport_cost']
            ep_costs['spoilage']  += info['spoilage_cost']
            ep_costs['stockout']  += info['stockout_cost']
            route_counts[action]  += 1

        results.append({'reward': ep_reward, **ep_costs, 'route_counts': route_counts})

    env.close()
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_learning_curve(reward_logger: RewardLoggerCallback):
    rewards = reward_logger.episode_rewards
    if not rewards:
        print("  No episode rewards logged — skipping learning curve.")
        return

    window = max(1, len(rewards) // 40)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards,  alpha=0.25, color='steelblue', linewidth=0.8, label='Episode reward')
    ax.plot(range(window - 1, len(rewards)), smoothed,
            color='steelblue', linewidth=2.0, label=f'Smoothed (window={window})')
    ax.axhline(np.mean(rewards[-len(rewards)//5:]), color='firebrick',
               linestyle='--', linewidth=1.2, label='Last-20% mean')
    ax.set_title('PPO Training — Learning Curve', fontsize=13, fontweight='bold')
    ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'rl_learning_curve.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: rl_learning_curve.png")


def plot_policy_comparison(random_results, ppo_results):
    labels  = ['Transport', 'Spoilage', 'Stockout']
    r_means = [np.mean([r['transport'] for r in random_results]),
               np.mean([r['spoilage']  for r in random_results]),
               np.mean([r['stockout']  for r in random_results])]
    p_means = [np.mean([r['transport'] for r in ppo_results]),
               np.mean([r['spoilage']  for r in ppo_results]),
               np.mean([r['stockout']  for r in ppo_results])]

    x = np.arange(len(labels))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Cost breakdown
    axes[0].bar(x - w/2, r_means, w, label='Random Policy', color='tomato',    alpha=0.8)
    axes[0].bar(x + w/2, p_means, w, label='PPO Policy',    color='steelblue', alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_title('Average Cost per Episode: Random vs PPO', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Cost (normalised)'); axes[0].legend()
    for xi, rv, pv in zip(x, r_means, p_means):
        pct = (rv - pv) / max(rv, 1e-9) * 100
        axes[0].text(xi + w/2, pv + 0.01, f'−{pct:.0f}%', ha='center',
                     fontsize=8, color='steelblue', fontweight='bold')

    # Total reward distribution
    r_rewards = [r['reward'] for r in random_results]
    p_rewards = [r['reward'] for r in ppo_results]
    axes[1].hist(r_rewards, bins=20, alpha=0.6, color='tomato',    label=f'Random  μ={np.mean(r_rewards):.1f}')
    axes[1].hist(p_rewards, bins=20, alpha=0.6, color='steelblue', label=f'PPO     μ={np.mean(p_rewards):.1f}')
    axes[1].axvline(np.mean(r_rewards), color='firebrick',  linestyle='--', linewidth=1.5)
    axes[1].axvline(np.mean(p_rewards), color='navy',       linestyle='--', linewidth=1.5)
    axes[1].set_title('Episode Reward Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Total Episode Reward'); axes[1].set_ylabel('Frequency')
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'rl_policy_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: rl_policy_comparison.png")


def plot_route_heatmap(ppo_results):
    route_matrix = np.zeros(15)
    for r in ppo_results:
        route_matrix += r['route_counts']
    route_matrix /= route_matrix.sum()

    env    = ColdChainRoutingEnv()
    labels = [f"{env.WAREHOUSE_NAMES[r['pair'][0]][:3]}↔{env.WAREHOUSE_NAMES[r['pair'][1]][:3]}"
              for r in env.ROUTES]
    env.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, route_matrix, color='steelblue', alpha=0.8)
    ax.set_title('PPO Route Preference Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Selection Frequency'); ax.set_xlabel('Route')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.axhline(1/15, color='firebrick', linestyle='--', linewidth=1, label='Uniform (random)')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'rl_route_preferences.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: rl_route_preferences.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR,   exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Sanity-check environment ──────────────────────────────────────────
    print("Verifying environment…")
    env = ColdChainRoutingEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (28,), f"Unexpected obs shape: {obs.shape}"
    obs, reward, term, trunc, info = env.step(0)
    print(f"  Obs shape : {obs.shape}")
    print(f"  Sample reward: {reward:.4f}  |  {info}")
    env.close()

    # ── Baseline: random policy ───────────────────────────────────────────
    print("\nBaseline: random policy (50 episodes)…")
    random_results = run_episodes('random', n_episodes=50)
    r_mean = np.mean([r['reward'] for r in random_results])
    print(f"  Random mean reward: {r_mean:.3f}")

    # ── Train PPO ─────────────────────────────────────────────────────────
    print(f"\nTraining PPO for {TOTAL_TIMESTEPS:,} timesteps…")
    vec_env   = make_vec_env(ColdChainRoutingEnv, n_envs=4, seed=42)
    eval_env  = ColdChainRoutingEnv()

    reward_logger = RewardLoggerCallback()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, 'best_model'),
        log_path=LOG_DIR,
        eval_freq=5_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=0,
    )

    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate   = 3e-4,
        n_steps         = 512,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.20,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        verbose         = 1,
        seed            = 42,
    )
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [reward_logger, eval_callback],
        progress_bar    = False,
    )
    model.save(MODEL_OUT)
    print(f"\n  Model saved → {MODEL_OUT}.zip")

    vec_env.close()
    eval_env.close()

    # ── Evaluate PPO ──────────────────────────────────────────────────────
    print("\nEvaluating trained PPO (50 episodes)…")
    ppo_results = run_episodes(model, n_episodes=50, deterministic=True)
    p_mean = np.mean([r['reward'] for r in ppo_results])
    improvement = (p_mean - r_mean) / abs(r_mean) * 100
    print(f"  PPO mean reward   : {p_mean:.3f}")
    print(f"  Improvement vs random: {improvement:+.1f}%")

    spoil_r = np.mean([r['spoilage']  for r in random_results])
    spoil_p = np.mean([r['spoilage']  for r in ppo_results])
    stock_r = np.mean([r['stockout']  for r in random_results])
    stock_p = np.mean([r['stockout']  for r in ppo_results])
    print(f"  Spoilage cost  random={spoil_r:.3f}  ppo={spoil_p:.3f}  ({(spoil_p-spoil_r)/max(spoil_r,1e-9)*100:+.1f}%)")
    print(f"  Stockout cost  random={stock_r:.3f}  ppo={stock_p:.3f}  ({(stock_p-stock_r)/max(stock_r,1e-9)*100:+.1f}%)")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    plot_learning_curve(reward_logger)
    plot_policy_comparison(random_results, ppo_results)
    plot_route_heatmap(ppo_results)

    print("\nPhase 3 complete.")


if __name__ == '__main__':
    main()
