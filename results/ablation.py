"""
Reward-weight ablation — v2.

Strategy
--------
  1. Load the 2M pre-trained baseline (already converged general routing policy).
  2. Fine-tune each variant from that baseline for 1M steps with extreme weights.
     Warm-starting avoids the noise of cold-start random policies and produces
     cleaner behavioural contrasts.
  3. Evaluate all variants on a NEUTRAL environment (baseline weights) so the
     comparison is weight-independent.

Configs
-------
  baseline       C_spoilage=15   C_stockout=10   (2M model, no fine-tuning)
  spoilage_5x    C_spoilage=75   C_stockout=10   (5× spoilage — should cut breaches)
  spoilage_10x   C_spoilage=150  C_stockout=10   (10× spoilage — strongest cold-chain signal)
  stockout_5x    C_spoilage=15   C_stockout=50   (5× stockout — should cut unmet demand)

Expected result
---------------
  spoilage variants   → fewer breach degrees, higher stockout volume
  stockout variant    → lower stockout volume, higher breach degrees
  baseline            → sits between the extremes

If even 10× spoilage produces no behavioural shift, the limitation is structural
(state/action space), not the reward — itself a publishable finding.
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.rl_routing.environment import ColdChainRoutingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

SAVE_DIR      = os.path.join(os.path.dirname(__file__), '..', 'models', 'rl_routing')
PLOTS_DIR     = os.path.join(os.path.dirname(__file__), '..', 'models', 'plots')
SUMMARY_PATH  = os.path.join(os.path.dirname(__file__), 'summary.json')
BASELINE_PATH = os.path.join(SAVE_DIR, 'ppo_cold_chain')
FINETUNE_STEPS = 1_000_000
N_EVAL         = 100

CONFIGS = {
    'baseline':     {'c_spoilage':  15.0, 'c_stockout': 10.0},
    'spoilage_5x':  {'c_spoilage':  75.0, 'c_stockout': 10.0},
    'spoilage_10x': {'c_spoilage': 150.0, 'c_stockout': 10.0},
    'stockout_5x':  {'c_spoilage':  15.0, 'c_stockout': 50.0},
}

COLORS = {
    'baseline':     'steelblue',
    'spoilage_5x':  'tomato',
    'spoilage_10x': 'firebrick',
    'stockout_5x':  'goldenrod',
}
LABELS = {
    'baseline':     'Baseline\n(S=15, K=10)',
    'spoilage_5x':  'Spoilage 5×\n(S=75, K=10)',
    'spoilage_10x': 'Spoilage 10×\n(S=150, K=10)',
    'stockout_5x':  'Stockout 5×\n(S=15, K=50)',
}


# ── Train / load ──────────────────────────────────────────────────────────────

def load_or_finetune(name: str, cfg: dict) -> PPO:
    model_path = os.path.join(SAVE_DIR, f'ppo_ablation_{name}.zip')

    if os.path.exists(model_path):
        print(f"  [{name}] Loading cached model.")
        return PPO.load(model_path)

    if name == 'baseline':
        print(f"  [{name}] Loading 2M pre-trained baseline.")
        return PPO.load(BASELINE_PATH)

    print(f"  [{name}] Warm-starting from baseline, fine-tuning {FINETUNE_STEPS:,} steps "
          f"(C_spoilage={cfg['c_spoilage']}, C_stockout={cfg['c_stockout']})…")

    vec_env = make_vec_env(
        lambda cfg=cfg: ColdChainRoutingEnv(c_spoilage=cfg['c_spoilage'],
                                            c_stockout=cfg['c_stockout']),
        n_envs=4, seed=42,
    )
    # Load baseline weights into the new environment
    model = PPO.load(BASELINE_PATH, env=vec_env)
    model.learn(total_timesteps=FINETUNE_STEPS, reset_num_timesteps=True)
    model.save(model_path)
    vec_env.close()
    print(f"  [{name}] Saved → {model_path}")
    return model


# ── Evaluate on neutral environment ───────────────────────────────────────────

def evaluate_raw(model, n=N_EVAL) -> dict:
    env = ColdChainRoutingEnv()   # neutral / baseline weights
    breach_list, stockout_list, dist_list, spoil_frac_list = [], [], [], []

    for ep in range(n):
        obs, _ = env.reset(seed=ep)
        done   = False
        ep_breach = ep_stockout = ep_dist = ep_spoil_frac = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(int(action))
            done = term or trunc
            ep_breach      += info['raw_breach_deg_c']
            ep_stockout    += info['raw_stockout_units']
            ep_dist        += info['raw_distance_km']
            ep_spoil_frac  += info['spoilage_frac']
            steps += 1
        breach_list.append(ep_breach)
        stockout_list.append(ep_stockout)
        dist_list.append(ep_dist)
        spoil_frac_list.append(ep_spoil_frac / max(steps, 1))

    env.close()
    return {
        'breach_mean':     float(np.mean(breach_list)),
        'breach_std':      float(np.std(breach_list)),
        'stockout_mean':   float(np.mean(stockout_list)),
        'stockout_std':    float(np.std(stockout_list)),
        'distance_mean':   float(np.mean(dist_list)),
        'spoil_frac_mean': float(np.mean(spoil_frac_list)),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_ablation(results: dict):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    names  = list(results.keys())
    colors = [COLORS[n] for n in names]
    xlbls  = [LABELS[n]  for n in names]

    breach_means   = [results[n]['breach_mean']   for n in names]
    breach_stds    = [results[n]['breach_std']    for n in names]
    stockout_means = [results[n]['stockout_mean'] for n in names]
    stockout_stds  = [results[n]['stockout_std']  for n in names]
    spoil_means    = [results[n]['spoil_frac_mean'] * 100 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Reward-Weight Ablation: Agent Behaviour vs Cost Structure\n'
        '(all variants evaluated on neutral environment — comparison is weight-independent)',
        fontsize=12, fontweight='bold',
    )

    # ── Breach severity ───────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(xlbls, breach_means, color=colors, alpha=0.85,
                  yerr=breach_stds, capsize=5)
    ax.set_title('Breach Severity\n(°C above threshold, per episode)', fontsize=10)
    ax.set_ylabel('Total Breach °C')
    for bar, val in zip(bars, breach_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.axhline(results['baseline']['breach_mean'], color='steelblue',
               linestyle='--', linewidth=1, alpha=0.6)

    # ── Stockout volume ───────────────────────────────────────────────────
    ax = axes[1]
    bars = ax.bar(xlbls, stockout_means, color=colors, alpha=0.85,
                  yerr=stockout_stds, capsize=5)
    ax.set_title('Stockout Volume\n(unmet demand units, per episode)', fontsize=10)
    ax.set_ylabel('Total Stockout Units')
    for bar, val in zip(bars, stockout_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,.0f}', ha='center', fontsize=8, fontweight='bold')
    ax.axhline(results['baseline']['stockout_mean'], color='steelblue',
               linestyle='--', linewidth=1, alpha=0.6)

    # ── Pareto trade-off ──────────────────────────────────────────────────
    ax = axes[2]
    for name in names:
        ax.scatter(results[name]['breach_mean'], results[name]['stockout_mean'],
                   s=180, color=COLORS[name], zorder=5,
                   label=LABELS[name].replace('\n', ' '))
        ax.annotate(name.replace('_', '\n'),
                    (results[name]['breach_mean'], results[name]['stockout_mean']),
                    textcoords='offset points', xytext=(6, 4), fontsize=7.5)

    # Connect spoilage variants with an arrow to show the trade-off direction
    if 'spoilage_10x' in results and 'stockout_5x' in results:
        ax.annotate('',
            xy=(results['spoilage_10x']['breach_mean'],
                results['spoilage_10x']['stockout_mean']),
            xytext=(results['stockout_5x']['breach_mean'],
                    results['stockout_5x']['stockout_mean']),
            arrowprops=dict(arrowstyle='<->', color='grey', lw=1.5, linestyle='dashed'))

    ax.set_xlabel('Breach Severity (°C)'); ax.set_ylabel('Stockout Volume (units)')
    ax.set_title('Pareto Trade-off\n(Spoilage vs Stockout)', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, 'ablation_reward_weights.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Reward-Weight Ablation (warm-start, extreme weights) ===\n")

    models  = {}
    results = {}

    for name, cfg in CONFIGS.items():
        models[name] = load_or_finetune(name, cfg)

    print(f"\nEvaluating all variants ({N_EVAL} episodes, neutral weights)…")
    for name, model in models.items():
        results[name] = evaluate_raw(model)
        r = results[name]
        print(f"  {name:<14}  breach={r['breach_mean']:6.2f}°C  "
              f"stockout={r['stockout_mean']:8,.0f}  "
              f"spoil_frac={r['spoil_frac_mean']*100:.2f}%")

    # ── Behaviour shift table ─────────────────────────────────────────────
    base = results['baseline']
    print("\n── Behaviour shift vs baseline ──────────────────────────────────")
    print(f"  {'Variant':<14}  {'Breach Δ':>10}  {'Stockout Δ':>12}  Interpretation")
    print("  " + "─" * 65)
    for name in list(CONFIGS.keys())[1:]:
        db = (results[name]['breach_mean']   - base['breach_mean'])   / max(base['breach_mean'],   1e-9) * 100
        ds = (results[name]['stockout_mean'] - base['stockout_mean']) / max(base['stockout_mean'], 1e-9) * 100
        interp = ''
        if 'spoilage' in name:
            interp = '✓ less spoilage' if db < -2 else ('~ no shift' if abs(db) < 2 else '✗ more spoilage')
        else:
            interp = '✓ less stockout' if ds < -2 else ('~ no shift' if abs(ds) < 2 else '✗ more stockout')
        print(f"  {name:<14}  {db:>+9.1f}%  {ds:>+11.1f}%  {interp}")

    responsive = any(
        abs((results[n]['breach_mean'] - base['breach_mean']) / max(base['breach_mean'], 1e-9)) > 0.02
        or abs((results[n]['stockout_mean'] - base['stockout_mean']) / max(base['stockout_mean'], 1e-9)) > 0.02
        for n in list(CONFIGS.keys())[1:]
    )

    print("\n── Verdict ──────────────────────────────────────────────────────")
    if responsive:
        print("  ✓ Agent is responsive to reward weights.")
        print("    Policy adapts to cost structure — tunable for real deployment.")
    else:
        print("  ~ Behavioural shift is small — limitation is structural.")
        print("    The state/action space constrains what the policy can express,")
        print("    independent of reward weighting. This is itself a valid finding:")
        print("    architecture changes (richer state, finer action space) would")
        print("    unlock stronger specialisation.")

    plot_ablation(results)

    # ── Update summary.json ───────────────────────────────────────────────
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    summary['phase_3_rl_routing']['ablation'] = {
        'strategy': 'Warm-start from 2M baseline, fine-tune 1M steps per variant',
        'finetune_steps': FINETUNE_STEPS,
        'evaluation_episodes': N_EVAL,
        'evaluation_environment': 'neutral (baseline weights) — weight-independent comparison',
        'configs': CONFIGS,
        'results': {
            name: {
                'breach_deg_c_mean':    round(r['breach_mean'],     2),
                'breach_deg_c_std':     round(r['breach_std'],      2),
                'stockout_units_mean':  round(r['stockout_mean'],   1),
                'stockout_units_std':   round(r['stockout_std'],    1),
                'spoilage_frac_pct':    round(r['spoil_frac_mean'] * 100, 3),
            }
            for name, r in results.items()
        },
        'agent_responsive': responsive,
        'plot': 'models/plots/ablation_reward_weights.png',
    }

    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Updated → results/summary.json")
    print("\nAblation complete.")


if __name__ == '__main__':
    main()
