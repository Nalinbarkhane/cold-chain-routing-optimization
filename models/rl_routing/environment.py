import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ColdChainRoutingEnv(gym.Env):
    """
    Cold-chain truck routing environment.

    The agent dispatches one truck per step, choosing which warehouse-pair
    route to service.  Trucks auto-route from the higher-inventory warehouse
    to the lower-inventory one in the chosen pair.

    Observation (28-dim, all in [0, 1]):
        inventory_ratio  [6]  — current stock / max capacity, per warehouse
        demand_ratio     [6]  — today's demand / max demand, per warehouse
        weather_severity [6]  — 0=calm … 1=severe, per region
        stockout_risk    [6]  — 1 if inventory < today's demand, else 0
        cargo_category   [4]  — one-hot for the current shipment type

    Action:
        Discrete(15) — choose one of 15 warehouse-pair routes.

    Reward:
        R_t = -(c_transport * d_t
                + c_spoilage * max(0, T_truck - T_threshold)
                + c_stockout * S_missed)
    """

    metadata = {'render_modes': ['human']}

    # ── Cost coefficient defaults (overridable via __init__) ────────────────
    _DEFAULT_C_TRANSPORT = 1.0
    _DEFAULT_C_SPOILAGE  = 15.0
    _DEFAULT_C_STOCKOUT  = 10.0

    # ── Episode config ───────────────────────────────────────────────────────
    EPISODE_DAYS   = 30
    TRUCK_CAPACITY = 1_500.0   # units delivered per trip
    MAX_INVENTORY  = 25_000.0
    MAX_DEMAND     =  5_500.0
    MAX_DISTANCE   =  4_200.0  # km (Boston ↔ Seattle, approx max)

    # ── Warehouse definitions (index 0–5 = SQL IDs 1–6) ─────────────────────
    WAREHOUSE_NAMES = ['Boston', 'Atlanta', 'Chicago', 'Dallas', 'LA', 'Seattle']
    REGIONS         = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West', 'Northwest']
    BASE_DEMAND     = np.array([3_500., 3_300., 4_600., 3_100., 4_400., 2_700.], dtype=np.float32)

    # ── 15 undirected routes (all warehouse pairs) ───────────────────────────
    ROUTES = [
        {'pair': (0, 1), 'distance_km': 1_505},   # Boston  ↔ Atlanta
        {'pair': (0, 2), 'distance_km': 1_371},   # Boston  ↔ Chicago
        {'pair': (0, 3), 'distance_km': 2_758},   # Boston  ↔ Dallas
        {'pair': (0, 4), 'distance_km': 4_168},   # Boston  ↔ LA
        {'pair': (0, 5), 'distance_km': 4_171},   # Boston  ↔ Seattle
        {'pair': (1, 2), 'distance_km': 1_147},   # Atlanta ↔ Chicago
        {'pair': (1, 3), 'distance_km': 1_157},   # Atlanta ↔ Dallas
        {'pair': (1, 4), 'distance_km': 3_107},   # Atlanta ↔ LA
        {'pair': (1, 5), 'distance_km': 3_617},   # Atlanta ↔ Seattle
        {'pair': (2, 3), 'distance_km': 1_504},   # Chicago ↔ Dallas
        {'pair': (2, 4), 'distance_km': 2_808},   # Chicago ↔ LA
        {'pair': (2, 5), 'distance_km': 2_649},   # Chicago ↔ Seattle
        {'pair': (3, 4), 'distance_km': 1_993},   # Dallas  ↔ LA
        {'pair': (3, 5), 'distance_km': 2_669},   # Dallas  ↔ Seattle
        {'pair': (4, 5), 'distance_km': 1_544},   # LA      ↔ Seattle
    ]

    # ── Cargo categories (temp in °C) ─────────────────────────────────────────
    CARGO = [
        {'name': 'pharmaceutical', 'target_temp':   5.0, 'threshold':   8.0},
        {'name': 'dairy',          'target_temp':   2.5, 'threshold':   4.0},
        {'name': 'produce',        'target_temp':   1.5, 'threshold':   4.0},
        {'name': 'frozen',         'target_temp': -18.0, 'threshold': -15.0},
    ]

    # ── Gym spaces ────────────────────────────────────────────────────────────
    def __init__(self, c_transport=None, c_spoilage=None, c_stockout=None):
        super().__init__()
        self.C_TRANSPORT = c_transport if c_transport is not None else self._DEFAULT_C_TRANSPORT
        self.C_SPOILAGE  = c_spoilage  if c_spoilage  is not None else self._DEFAULT_C_SPOILAGE
        self.C_STOCKOUT  = c_stockout  if c_stockout  is not None else self._DEFAULT_C_STOCKOUT
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(28,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.ROUTES))

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0

        # Initial inventory: 2–4 days of demand per warehouse
        self.inventory = (
            self.BASE_DEMAND * self.np_random.uniform(2.0, 4.0, size=6)
        ).astype(np.float32)

        # Stochastic demand schedule for the episode (±30 % noise around base)
        noise = self.np_random.uniform(0.70, 1.30, size=(self.EPISODE_DAYS, 6))
        self.demand_schedule = (self.BASE_DEMAND * noise).astype(np.float32)

        # Weather per region: AR(1) process, starts mildly random
        self.weather = self.np_random.uniform(0.0, 0.4, size=6).astype(np.float32)

        # Random cargo for first step
        self.cargo_idx = int(self.np_random.integers(0, len(self.CARGO)))

        return self._get_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action: int):
        route  = self.ROUTES[action]
        a, b   = route['pair']
        dist   = route['distance_km']
        cargo  = self.CARGO[self.cargo_idx]
        today_demand = self.demand_schedule[self.day]

        # Determine direction: ship from higher-inventory end to lower
        origin, dest = (a, b) if self.inventory[a] >= self.inventory[b] else (b, a)

        # Simulate truck temperature (mean-reverting walk + weather spike)
        route_weather  = float(max(self.weather[origin], self.weather[dest]))
        noise          = self.np_random.normal(0, 0.8 + route_weather * 1.5)
        if self.np_random.random() < 0.05 + route_weather * 0.15:   # spike probability
            noise += float(self.np_random.uniform(2.0, 6.0))
        truck_temp     = cargo['target_temp'] + noise

        # Spoilage fraction: linear from threshold to 5°C above it
        breach         = max(0.0, truck_temp - cargo['threshold'])
        spoil_frac     = min(1.0, breach / 5.0)

        # Delivery: goods spoil proportionally
        shipped    = min(float(self.inventory[origin]), self.TRUCK_CAPACITY)
        delivered  = shipped * (1.0 - spoil_frac)

        # Update inventory
        self.inventory[origin] = max(0.0, self.inventory[origin] - shipped)
        self.inventory[dest]   = min(self.MAX_INVENTORY, self.inventory[dest] + delivered)

        # Daily demand consumption at all warehouses
        self.inventory = np.maximum(0.0, self.inventory - today_demand * 0.4)

        # Stockout at destination after consumption
        stockout_units = max(0.0, today_demand[dest] - self.inventory[dest])

        # ── Reward function ────────────────────────────────────────────────
        r_transport = self.C_TRANSPORT * (dist / self.MAX_DISTANCE)
        r_spoilage  = self.C_SPOILAGE  * breach
        r_stockout  = self.C_STOCKOUT  * (stockout_units / self.MAX_DEMAND)
        reward      = float(-(r_transport + r_spoilage + r_stockout))

        # Evolve weather (slow AR-1 drift toward 0.3 baseline)
        self.weather = np.clip(
            0.95 * self.weather + 0.05 * 0.3
            + self.np_random.normal(0, 0.05, size=6),
            0.0, 1.0
        ).astype(np.float32)

        # Next cargo
        self.cargo_idx = int(self.np_random.integers(0, len(self.CARGO)))
        self.day += 1

        terminated = self.day >= self.EPISODE_DAYS
        truncated  = False
        info = {
            'transport_cost':  r_transport,
            'spoilage_cost':   r_spoilage,
            'stockout_cost':   r_stockout,
            'truck_temp_c':    round(truck_temp, 2),
            'breach_deg_c':    round(breach, 2),
            'delivered_units': round(delivered, 1),
            'spoilage_frac':   round(spoil_frac, 3),
            'cargo':           cargo['name'],
            'route':           f"{self.WAREHOUSE_NAMES[origin]} → {self.WAREHOUSE_NAMES[dest]}",
            # Raw (unweighted) metrics — used for fair cross-config comparison
            'raw_breach_deg_c':    round(breach, 2),
            'raw_stockout_units':  round(stockout_units, 1),
            'raw_distance_km':     dist,
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ── Observation ───────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        day_idx      = min(self.day, self.EPISODE_DAYS - 1)
        demand       = self.demand_schedule[day_idx]
        inv_ratio    = np.clip(self.inventory / self.MAX_INVENTORY, 0.0, 1.0)
        dem_ratio    = np.clip(demand / self.MAX_DEMAND, 0.0, 1.0)
        stockout     = (inv_ratio < dem_ratio).astype(np.float32)
        cargo_onehot = np.eye(len(self.CARGO), dtype=np.float32)[self.cargo_idx]
        return np.concatenate([inv_ratio, dem_ratio, self.weather, stockout, cargo_onehot])

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self):
        print(f"\n── Day {self.day}/{self.EPISODE_DAYS} ──")
        for i, name in enumerate(self.WAREHOUSE_NAMES):
            inv = self.inventory[i]
            dem = self.demand_schedule[min(self.day, self.EPISODE_DAYS-1)][i]
            wth = self.weather[i]
            print(f"  {name:<10}  inv={inv:7.0f}  demand={dem:6.0f}  weather={wth:.2f}")
