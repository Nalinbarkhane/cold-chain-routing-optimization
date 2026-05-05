# Intelligent Cold-Chain Routing & Dynamic Inventory Optimization

An end-to-end AI supply-chain system for temperature-sensitive goods (pharmaceuticals, dairy, produce, frozen). The pipeline moves from raw data ingestion through predictive demand forecasting to a prescriptive reinforcement-learning routing agent — covering the full analytics maturity curve in a single project.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER (Dual-DB)                          │
│                                                                      │
│   ┌────────────────────────┐        ┌─────────────────────────────┐  │
│   │      PostgreSQL         │        │          MongoDB             │  │
│   │   Structured MDM        │        │   High-velocity Telemetry    │  │
│   │                         │        │                              │  │
│   │  warehouses       (6)   │        │  iot_telemetry   (29,232)    │  │
│   │  skus            (12)   │        │  weather_events     (450)    │  │
│   │  daily_demand  (52,632) │        │  supplier_delays    (200)    │  │
│   │  inventory     (52,632) │        │                              │  │
│   │  routes          (15)   │        │  GPS · temp · humidity       │  │
│   │                         │        │  from 8 simulated trucks     │  │
│   └───────────┬─────────────┘        └──────────────┬──────────────┘  │
│               │                                      │                 │
│               └──────────────────┬───────────────────┘                 │
│                                  │                                     │
│                        ┌─────────▼──────────┐                         │
│                        │    ETL Pipeline      │                         │
│                        │   etl/pipeline.py    │                         │
│                        │  52,632 rows × 34   │                         │
│                        │   feature columns   │                         │
│                        └─────────┬──────────┘                         │
└──────────────────────────────────┼─────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
   ┌──────────▼──────────┐                  ┌──────────▼──────────┐
   │      PHASE 2         │                  │      PHASE 3         │
   │  XGBoost Forecaster  │  demand signal   │   PPO Routing Agent  │
   │  models/demand_      ├─────────────────▶│   models/rl_routing/ │
   │  forecast.py         │                  │   train_agent.py     │
   │                      │                  │                      │
   │  R²    =  0.978       │                  │  +19.3% vs greedy    │
   │  MAPE  =  7.37 %      │                  │  +14.7% vs random    │
   │  MAE   = 22.32 units  │                  │  Reward tunable to   │
   └──────────────────────┘                  │  business economics  │
                                             └──────────────────────┘
```

---

## The Problem

Managing supply chains for temperature-sensitive goods requires balancing two competing risks simultaneously:

- **Stockouts** — running out of inventory at a warehouse because demand was underestimated or deliveries were delayed
- **Spoilage** — cargo being ruined in transit because a truck's cooling unit fluctuated or a route passed through severe weather

A purely rule-based system can't optimise both. This project builds an AI pipeline that (1) predicts demand accurately so you know *what* to stock, and (2) routes trucks dynamically so you know *how* to deliver it without spoilage.

---

## Phase 1 — Data Engineering

### Why Two Databases?

| Concern | PostgreSQL | MongoDB |
|---|---|---|
| Data type | Structured, relational | Semi-structured JSON |
| Velocity | Static master data | High-frequency IoT streams |
| Contents | Warehouses, SKUs, historical demand | GPS pings, temperature readings, weather events |
| Query pattern | Aggregations, joins | Time-series lookups, document scans |

PostgreSQL is the **backbone** — it holds the historical record of what happened. MongoDB is the **nervous system** — it ingests real-time signals that change what should happen next.

### What Was Built

```
data_engineering/sql_setup.py       → Creates schema, seeds 2 years of demand data
data_engineering/mongo_streaming.py → Simulates IoT truck telemetry + weather + supplier delays
etl/pipeline.py                     → Joins both sources into a 34-feature matrix
```

**Scale:** 52,632 demand records · 29,232 IoT documents · 450 weather events · 200 supplier delays  
**Date range:** 2023-01-01 → 2024-12-31  
**ETL output:** `data/features.csv` — 52,632 rows × 34 columns

---

## Phase 2 — Predictive Demand Forecasting

### Model

XGBoost regressor trained on the ETL feature matrix with a time-based train/val/test split.

| Split | Period | Rows |
|---|---|---|
| Train | Jan 2023 – Jul 2024 | 39,600 |
| Validation (early stopping) | Aug – Sep 2024 | 4,392 |
| Test (holdout) | Oct – Dec 2024 | 6,624 |

### Results

| Metric | Value |
|---|---|
| MAE | **22.32 units** |
| RMSE | **33.90 units** |
| MAPE | **7.37 %** |
| R² | **0.978** |

The model converged at **283 trees** (early-stopped from 1,000). An R² of 0.978 means it explains 97.8 % of demand variance on unseen data.

### Feature Groups

The 34 columns fed to XGBoost span four categories:

1. **Temporal** — day-of-week, month, quarter, week-of-year, is-weekend
2. **Lag features** — demand 7, 14, and 28 days prior (within each SKU × warehouse series)
3. **Rolling statistics** — 7/14/28-day rolling mean and standard deviation
4. **External signals** — weather severity score, IoT breach count (prior day), supplier delay hours

Rolling means and lags dominate the feature importance ranking — the model learns that recent history is the strongest predictor of near-future demand.

### Plots

| Plot | Description |
|---|---|
| [Feature importance](models/plots/feature_importance.png) | Top-20 XGBoost gain scores |
| [Actual vs predicted](models/plots/actual_vs_predicted.png) | Scatter across all categories on test set |
| [Time-series forecast](models/plots/time_series_forecast.png) | 4 SKU/warehouse combos showing seasonal capture |
| [Error distribution](models/plots/error_distribution.png) | Residual histogram + MAPE by category |

---

## Phase 3 — Prescriptive RL Routing Agent

### Environment

A custom Gymnasium environment (`ColdChainRoutingEnv`) where the agent dispatches one truck per step and learns to minimise the combined cost of transport, spoilage, and stockouts.

```
Observation space  Box(28,)  — inventory ratio (6), demand ratio (6),
                               weather severity (6), stockout risk (6),
                               cargo category one-hot (4)

Action space       Discrete(15) — choose one of 15 warehouse-pair routes
                               (truck auto-routes from higher to lower inventory end)

Episode length     30 days
```

### Reward Function

$$R_t = -\bigl(c_{\text{transport}} \cdot d_t + c_{\text{spoilage}} \cdot \max(0,\, T_{\text{truck}} - T_{\text{threshold}}) + c_{\text{stockout}} \cdot S_{\text{missed}}\bigr)$$

| Coefficient | Value | Penalises |
|---|---|---|
| $c_{\text{transport}}$ | 1.0 | Route distance |
| $c_{\text{spoilage}}$ | 15.0 | Degrees above cargo temperature threshold |
| $c_{\text{stockout}}$ | 10.0 | Unmet demand at destination warehouse |

### Training

PPO (Proximal Policy Optimization) via Stable-Baselines3, trained for **2 million timesteps** across 4 parallel environments. Ablation variants were warm-started from this checkpoint and fine-tuned for an additional 1 million timesteps each.

### Policy Comparison (100-episode evaluation)

| Policy | Mean Reward | vs Random | vs Greedy |
|---|---|---|---|
| Random | −265.5 | — | — |
| Greedy heuristic | −280.7 | −5.7 % | — |
| **PPO agent** | **−226.5** | **+14.7 %** | **+19.3 %** |

**Key finding:** The greedy heuristic (always route to the most under-stocked warehouse via the shortest path) actually *loses* to random — it optimises short routes but triggers stockout cascades elsewhere. PPO beats both by learning to balance all three cost terms simultaneously.

### Cost Breakdown — PPO Agent

| Cost term | Mean per episode | % of total penalty |
|---|---|---|
| Transport | 13.0 | 5.7 % |
| Spoilage | 92.3 | 40.7 % |
| Stockout | 121.2 | 53.5 % |

All three cost terms contribute meaningfully to the total penalty. Stockout is the dominant signal the agent optimises against, with spoilage and transport as secondary considerations.

### Plots

| Plot | Description |
|---|---|
| [Learning curve](models/plots/rl_learning_curve.png) | Smoothed episode reward rising from ~−260 → ~−235 over ~67,000 training episodes (2M timesteps); last-20% mean ≈ −235 |
| [Policy comparison](models/plots/rl_policy_comparison.png) | Random vs PPO cost breakdown |
| [Route preferences](models/plots/rl_route_preferences.png) | Which routes the agent favours vs uniform random |
| [Diagnosis comparison](models/plots/diagnosis_comparison.png) | Random vs Greedy vs PPO side-by-side |

---

## Ablation Study — Reward Weight Sensitivity

To verify that the agent is *responsive to the objective you specify*, three variants were fine-tuned from the baseline using extreme weight changes and evaluated on a neutral (baseline-weight) environment so the comparison is weight-independent.

| Variant | $c_{\text{spoilage}}$ | $c_{\text{stockout}}$ | Breach °C | Stockout units |
|---|---|---|---|---|
| Baseline | 15 | 10 | 5.87 | 66,744 |
| Spoilage 5× | 75 | 10 | **5.75** | 69,676 |
| Spoilage 10× | 150 | 10 | 6.03 | 66,888 |
| Stockout 5× | 15 | 50 | 6.55 | **64,381** |

**Interpretation:** At moderate weights (5×), the agent demonstrates clear cost-driven trade-offs — up-weighting stockout cost reduces unmet demand by ~3,600 units at the cost of more temperature exposure, while up-weighting spoilage reduces breach with a corresponding rise in stockouts. Notably, the Spoilage 5× variant Pareto-dominates the baseline on temperature compliance with minimal stockout penalty, suggesting the original weight calibration may have under-prioritised spoilage. At extreme weights (10× spoilage), the relationship becomes non-monotonic — breach actually increases slightly above baseline — suggesting that policy optimisation is sensitive to reward scaling at these magnitudes. In deployment, cost weights can be tuned to match real business economics (pharmaceutical distributors would weight spoilage higher than frozen-food distributors), though production use would benefit from reward normalisation to ensure stable behaviour across the full operating range.

[Ablation plot](models/plots/ablation_reward_weights.png)

---

## Reproduce

### Prerequisites

**Platform notes:** Instructions below are for macOS (Homebrew). On Linux, replace `brew services start` with `sudo systemctl start postgresql` / `mongod --fork`. On Windows, start both services via the respective installer GUIs or service manager, then use `python` instead of `python3`.

```bash
# Start databases (macOS)
brew services start postgresql@14
brew services start mongodb-community

# Install Python dependencies — use the same python3 binary you will run scripts with
python3 -m pip install psycopg2-binary pymongo pandas numpy xgboost \
        scikit-learn matplotlib seaborn gymnasium stable-baselines3
brew install libomp   # macOS only — required by XGBoost
```

### Run the full pipeline

```bash
# 1 — Seed PostgreSQL (schema + 2 years of demand data)
python3 data_engineering/sql_setup.py

# 2 — Stream IoT telemetry + weather + supplier delays into MongoDB
python3 data_engineering/mongo_streaming.py

# 3 — ETL: join both databases → data/features.csv
python3 etl/pipeline.py

# 4 — Train XGBoost demand forecaster → models/demand_forecast.json
python3 models/demand_forecast.py

# 5 — Train PPO routing agent → models/rl_routing/ppo_cold_chain.zip
python3 models/rl_routing/train_agent.py

# 6 — Diagnose: Random vs Greedy vs PPO
python3 results/diagnose.py

# 7 — Ablation: reward-weight sensitivity
python3 results/ablation.py
```

Each script is self-contained and prints progress to stdout. Approximate wall-clock time on a modern laptop (M-series Mac or equivalent):

| Step | Time |
|---|---|
| Steps 1–3 (data + ETL) | ~2 min |
| Step 4 (XGBoost) | ~1 min |
| Step 5 (PPO 2M steps, 4 envs) | ~20–30 min |
| Steps 6–7 (diagnose + ablation) | ~10 min |

All random seeds are fixed (`seed=42` for training, `seed=ep` per evaluation episode) — results should be reproducible across runs on the same hardware.

---

## Limitations & Future Work

**Current limitations:**

- The RL environment dispatches exactly one truck per time-step. A multi-truck extension would require a multi-agent formulation or a fleet-aware action space.
- All data is synthetic. Demand seasonality, weather, and IoT breach rates are parameterised approximations — results on real logistics data would likely differ.
- The observation space does not include route transit times or current in-transit cargo, which limits the agent's ability to reason about delivery pipeline state.
- Reward normalisation is absent: at extreme weight multipliers (10× spoilage) the policy optimisation becomes unstable, as reflected in the non-monotonic ablation result.

**Natural next steps:**

- Integrate the XGBoost demand forecast directly into the RL observation so the agent acts on predicted future demand rather than today's snapshot.
- Replace discrete route selection with a continuous fleet-scheduling action space to support simultaneous multi-truck dispatch.
- Add reward normalisation (e.g. running mean/std scaling) to stabilise training under extreme cost configurations.
- Validate against a real cold-chain dataset (e.g. FDA pharmaceutical distribution records or public logistics benchmarks).

---

## Repository Structure

```
├── data_engineering/
│   ├── sql_setup.py          PostgreSQL schema + seed data
│   └── mongo_streaming.py    IoT telemetry + weather + supplier delays
├── etl/
│   └── pipeline.py           Join SQL + MongoDB → feature matrix
├── models/
│   ├── demand_forecast.py    XGBoost training + evaluation
│   ├── demand_forecast.json  Saved XGBoost model
│   ├── rl_routing/
│   │   ├── environment.py    Custom Gymnasium env (configurable reward)
│   │   ├── train_agent.py    PPO training + comparison plots
│   │   └── ppo_cold_chain.zip Saved PPO policy
│   └── plots/                All generated visualisations (9 PNG files)
├── data/
│   └── features.csv          ETL output — 52,632 × 34 feature matrix
├── results/
│   ├── summary.json          All metrics across all phases
│   ├── diagnose.py           3-policy comparison + scale check
│   └── ablation.py           Reward-weight sensitivity study
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Structured database | PostgreSQL 14 |
| Document / telemetry database | MongoDB 8.2 |
| Data manipulation | pandas, NumPy |
| Demand forecasting | XGBoost 3.x, scikit-learn |
| RL environment | Gymnasium 1.x |
| RL training | Stable-Baselines3 2.x (PPO) |
| Visualisation | Matplotlib, Seaborn |
| Language | Python 3.11 |

---

*Built by Nalin Barkhane — barkhane.n@northeastern.edu*
