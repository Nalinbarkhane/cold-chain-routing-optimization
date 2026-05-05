import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import math
from datetime import date, timedelta
import random
import itertools

random.seed(42)
np.random.seed(42)

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'cold_chain',
    'user': 'nalin',
}

WAREHOUSES = [
    (1, 'Boston Hub',     'Boston',      'MA', 'Northeast',  42.3601,  -71.0589, 50_000),
    (2, 'Atlanta Hub',    'Atlanta',     'GA', 'Southeast',  33.7490,  -84.3880, 45_000),
    (3, 'Chicago Hub',    'Chicago',     'IL', 'Midwest',    41.8781,  -87.6298, 60_000),
    (4, 'Dallas Hub',     'Dallas',      'TX', 'Southwest',  32.7767,  -96.7970, 40_000),
    (5, 'LA Hub',         'Los Angeles', 'CA', 'West',       34.0522, -118.2437, 55_000),
    (6, 'Seattle Hub',    'Seattle',     'WA', 'Northwest',  47.6062, -122.3321, 35_000),
]

SKUS = [
    #  id  name                    category          t_min   t_max  cost   shelf
    (1,  'Insulin Vials',         'pharmaceutical',   2.0,   8.0,  45.00,  30),
    (2,  'Flu Vaccines',          'pharmaceutical',   2.0,   8.0,  18.50,  90),
    (3,  'Blood Products',        'pharmaceutical',   1.0,   6.0, 120.00,   7),
    (4,  'Chemo Drugs',           'pharmaceutical',   2.0,   8.0, 350.00,  60),
    (5,  'Whole Milk',            'dairy',            1.0,   4.0,   1.20,  14),
    (6,  'Aged Cheddar',          'dairy',            2.0,   7.0,   4.50,  60),
    (7,  'Greek Yogurt',          'dairy',            1.0,   4.0,   2.80,  21),
    (8,  'Leafy Greens Mix',      'produce',          0.0,   4.0,   3.20,   7),
    (9,  'Mixed Berries',         'produce',          0.0,   2.0,   5.50,   5),
    (10, 'Frozen Entrees',        'frozen',         -20.0, -15.0,   6.00, 365),
    (11, 'Premium Ice Cream',     'frozen',         -22.0, -18.0,   5.50, 365),
    (12, 'Fresh Orange Juice',    'beverage',         1.0,   4.0,   3.80,  21),
]

# Base daily demand units per SKU (before any scaling)
BASE_DEMAND = {
    1: 200,  2: 150,  3: 80,   4: 50,
    5: 800,  6: 400,  7: 600,
    8: 500,  9: 300,
    10: 700, 11: 450, 12: 550,
}

# Warehouse scale factors (relative population / market size)
WH_SCALE = {1: 0.90, 2: 0.85, 3: 1.20, 4: 0.80, 5: 1.15, 6: 0.70}

# Seasonal (amplitude, phase_radians)
# Phase tuned: SUMMER_PEAK ~Jul (day 182), WINTER_PEAK ~Jan (day 15)
_S, _W = -1.4, 1.3
SEASONAL = {
    1:  (0.05, _W),   # Insulin — nearly flat, small winter uptick
    2:  (0.45, _W),   # Flu Vaccines — strong winter peak
    3:  (0.05, _S),   4:  (0.05, _S),   # Blood / Chemo — flat
    5:  (0.15, _S),   6:  (0.10, _S),   7:  (0.20, _S),   # Dairy — summer lean
    8:  (0.40, _S),   9:  (0.50, _S),   # Produce — strong summer peak
    10: (0.15, _W),   # Frozen entrees — slight winter (comfort food)
    11: (0.40, _S),   # Ice cream — summer peak
    12: (0.20, _W),   # OJ — winter peak (vitamins)
}

# Day-of-week multipliers: Mon(0) … Sun(6)
DOW_MULT = [1.10, 1.15, 1.05, 1.00, 0.95, 0.70, 0.65]


# ── Schema ────────────────────────────────────────────────────────────────────

def create_tables(conn):
    cur = conn.cursor()
    cur.execute("""
        DROP TABLE IF EXISTS inventory_levels, daily_demand,
                             transportation_routes, skus, warehouses CASCADE;

        CREATE TABLE warehouses (
            warehouse_id   SERIAL PRIMARY KEY,
            name           VARCHAR(60)      NOT NULL,
            city           VARCHAR(60)      NOT NULL,
            state_code     CHAR(2)          NOT NULL,
            region         VARCHAR(30)      NOT NULL,
            latitude       DOUBLE PRECISION,
            longitude      DOUBLE PRECISION,
            capacity_units INTEGER
        );

        CREATE TABLE skus (
            sku_id          SERIAL PRIMARY KEY,
            name            VARCHAR(80)      NOT NULL,
            category        VARCHAR(30)      NOT NULL,
            temp_min_c      DOUBLE PRECISION,
            temp_max_c      DOUBLE PRECISION,
            unit_cost       NUMERIC(10,2),
            shelf_life_days INTEGER
        );

        CREATE TABLE transportation_routes (
            route_id             SERIAL PRIMARY KEY,
            origin_warehouse_id  INTEGER REFERENCES warehouses(warehouse_id),
            dest_warehouse_id    INTEGER REFERENCES warehouses(warehouse_id),
            distance_km          DOUBLE PRECISION,
            base_cost_per_km     NUMERIC(6,2),
            avg_transit_hours    DOUBLE PRECISION
        );

        CREATE TABLE daily_demand (
            demand_id       BIGSERIAL PRIMARY KEY,
            sku_id          INTEGER REFERENCES skus(sku_id),
            warehouse_id    INTEGER REFERENCES warehouses(warehouse_id),
            date            DATE    NOT NULL,
            demand_units    INTEGER NOT NULL,
            fulfilled_units INTEGER NOT NULL,
            stockout_flag   BOOLEAN NOT NULL DEFAULT FALSE
        );
        CREATE INDEX idx_demand_date        ON daily_demand(date);
        CREATE INDEX idx_demand_sku_wh_date ON daily_demand(sku_id, warehouse_id, date);

        CREATE TABLE inventory_levels (
            inventory_id   BIGSERIAL PRIMARY KEY,
            sku_id         INTEGER REFERENCES skus(sku_id),
            warehouse_id   INTEGER REFERENCES warehouses(warehouse_id),
            date           DATE    NOT NULL,
            units_on_hand  INTEGER NOT NULL,
            units_on_order INTEGER NOT NULL DEFAULT 0,
            reorder_point  INTEGER NOT NULL
        );
        CREATE INDEX idx_inventory_sku_wh_date ON inventory_levels(sku_id, warehouse_id, date);
    """)
    conn.commit()
    print("Tables created.")


# ── Seed helpers ──────────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + (
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def _demand_units(sku_id, wh_id, day_of_year, weekday):
    base = BASE_DEMAND[sku_id] * WH_SCALE[wh_id]
    amp, phase = SEASONAL[sku_id]
    seasonal = 1 + amp * math.sin(2 * math.pi * day_of_year / 365 + phase)
    return max(1, round(base * seasonal * DOW_MULT[weekday] * np.random.normal(1.0, 0.08)))


# ── Insert functions ──────────────────────────────────────────────────────────

def insert_warehouses(conn):
    cur = conn.cursor()
    execute_values(cur,
        "INSERT INTO warehouses "
        "(warehouse_id, name, city, state_code, region, latitude, longitude, capacity_units) "
        "VALUES %s",
        WAREHOUSES,
    )
    conn.commit()
    print(f"  Inserted {len(WAREHOUSES)} warehouses.")


def insert_skus(conn):
    cur = conn.cursor()
    execute_values(cur,
        "INSERT INTO skus "
        "(sku_id, name, category, temp_min_c, temp_max_c, unit_cost, shelf_life_days) "
        "VALUES %s",
        SKUS,
    )
    conn.commit()
    print(f"  Inserted {len(SKUS)} SKUs.")


def insert_routes(conn):
    cur = conn.cursor()
    wh_map = {w[0]: w for w in WAREHOUSES}
    rows = []
    for a, b in itertools.combinations(range(1, 7), 2):
        wa, wb = wh_map[a], wh_map[b]
        dist    = _haversine_km(wa[5], wa[6], wb[5], wb[6])
        cost    = round(random.uniform(1.80, 3.50), 2)
        transit = round(dist / random.uniform(65, 85), 1)
        rows.append((a, b, round(dist, 1), cost, transit))
    execute_values(cur,
        "INSERT INTO transportation_routes "
        "(origin_warehouse_id, dest_warehouse_id, distance_km, base_cost_per_km, avg_transit_hours) "
        "VALUES %s",
        rows,
    )
    conn.commit()
    print(f"  Inserted {len(rows)} transportation routes.")


def insert_demand_and_inventory(conn):
    cur = conn.cursor()
    start, end = date(2023, 1, 1), date(2024, 12, 31)

    demand_rows, inventory_rows = [], []
    d = start
    while d <= end:
        doy = d.timetuple().tm_yday
        dow = d.weekday()
        for sku_id, *_, shelf_life in SKUS:
            for wh_id, *_ in WAREHOUSES:
                demand    = _demand_units(sku_id, wh_id, doy, dow)
                fulfilled = max(1, round(demand * random.uniform(0.88, 1.0)))
                stockout  = fulfilled < demand

                reorder_pt = round(BASE_DEMAND[sku_id] * WH_SCALE[wh_id] * 7)
                on_hand    = round(demand * random.uniform(0.9, 1.6))
                on_order   = (
                    max(0, round((reorder_pt - on_hand) * random.uniform(0.5, 1.5)))
                    if on_hand < reorder_pt else 0
                )

                demand_rows.append((sku_id, wh_id, d, demand, fulfilled, stockout))
                inventory_rows.append((sku_id, wh_id, d, on_hand, on_order, reorder_pt))

        d += timedelta(days=1)

    execute_values(cur,
        "INSERT INTO daily_demand "
        "(sku_id, warehouse_id, date, demand_units, fulfilled_units, stockout_flag) VALUES %s",
        demand_rows, page_size=5000,
    )
    execute_values(cur,
        "INSERT INTO inventory_levels "
        "(sku_id, warehouse_id, date, units_on_hand, units_on_order, reorder_point) VALUES %s",
        inventory_rows, page_size=5000,
    )
    conn.commit()
    print(f"  Inserted {len(demand_rows):,} demand records.")
    print(f"  Inserted {len(inventory_rows):,} inventory records.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Connecting to PostgreSQL (cold_chain)…")
    conn = psycopg2.connect(**DB_CONFIG)
    create_tables(conn)
    insert_warehouses(conn)
    insert_skus(conn)
    insert_routes(conn)
    insert_demand_and_inventory(conn)
    conn.close()
    print("\nSQL setup complete.")


if __name__ == '__main__':
    main()
