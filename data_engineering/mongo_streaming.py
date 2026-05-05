import psycopg2
from pymongo import MongoClient, ASCENDING
import numpy as np
from datetime import date, datetime, timedelta, time as dt_time
import random

random.seed(42)
np.random.seed(42)

PG_CONFIG   = {'host': 'localhost', 'port': 5432, 'dbname': 'cold_chain', 'user': 'nalin'}
MONGO_URI   = 'mongodb://localhost:27017/'
MONGO_DB    = 'cold_chain'

# Nominal operating temperature per SKU (°C)
SKU_TARGET_TEMP = {
    1: 5.0,  2: 5.0,  3: 3.5,  4: 5.0,   # pharmaceutical
    5: 2.5,  6: 4.5,  7: 2.5,             # dairy
    8: 2.0,  9: 1.0,                       # produce
    10: -17.5, 11: -20.0,                  # frozen
    12: 2.5,                               # beverage
}

REGIONS = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West', 'Northwest']

# Seasonally appropriate weather events per region
REGION_WEATHER = {
    'Northeast':  {'winter': ['blizzard', 'ice_storm'],        'summer': ['heat_wave', 'thunderstorm']},
    'Southeast':  {'winter': ['heavy_rain', 'fog'],             'summer': ['hurricane', 'heat_wave']},
    'Midwest':    {'winter': ['blizzard', 'ice_storm'],         'summer': ['tornado_watch', 'thunderstorm']},
    'Southwest':  {'winter': ['fog', 'heavy_rain'],             'summer': ['extreme_heat', 'dust_storm']},
    'West':       {'winter': ['mudslide', 'heavy_rain'],        'summer': ['wildfire_smoke', 'extreme_heat']},
    'Northwest':  {'winter': ['blizzard', 'heavy_rain'],        'summer': ['wildfire_smoke', 'heat_wave']},
}

SUPPLIERS = [
    {'id': 'SUP-001', 'name': 'PharmaCold Logistics',  'sku_ids': [1, 2, 3, 4]},
    {'id': 'SUP-002', 'name': 'DairyFresh Co.',         'sku_ids': [5, 6, 7]},
    {'id': 'SUP-003', 'name': 'GreenHarvest Inc.',      'sku_ids': [8, 9]},
    {'id': 'SUP-004', 'name': 'FrozenFoods Ltd.',       'sku_ids': [10, 11]},
    {'id': 'SUP-005', 'name': 'BeverageChill Corp.',    'sku_ids': [12]},
]

DELAY_REASONS = [
    'Production shutdown', 'Quality hold', 'Transportation disruption',
    'Raw material shortage', 'Equipment failure', 'Regulatory inspection',
    'Labor dispute', 'Packaging shortage', 'Cold storage failure',
]


# ── Collection setup ──────────────────────────────────────────────────────────

def setup_collections(db):
    for name in ['iot_telemetry', 'weather_events', 'supplier_delays']:
        if name in db.list_collection_names():
            db[name].drop()

    db.iot_telemetry.create_index([('truck_id', ASCENDING), ('timestamp', ASCENDING)])
    db.iot_telemetry.create_index([('timestamp', ASCENDING)])
    db.iot_telemetry.create_index([('temp_alert', ASCENDING)])
    db.iot_telemetry.create_index([('dest_warehouse_id', ASCENDING), ('timestamp', ASCENDING)])

    db.weather_events.create_index([('timestamp', ASCENDING)])
    db.weather_events.create_index([('region', ASCENDING), ('timestamp', ASCENDING)])

    db.supplier_delays.create_index([('timestamp', ASCENDING)])
    db.supplier_delays.create_index([('supplier_id', ASCENDING)])
    print("  Collections and indexes created.")


# ── Load routes from PostgreSQL ───────────────────────────────────────────────

def load_routes():
    conn = psycopg2.connect(**PG_CONFIG)
    cur  = conn.cursor()
    cur.execute("""
        SELECT r.route_id,
               r.origin_warehouse_id, r.dest_warehouse_id,
               r.avg_transit_hours,
               wo.latitude  AS o_lat, wo.longitude AS o_lon,
               wd.latitude  AS d_lat, wd.longitude AS d_lon
        FROM   transportation_routes r
        JOIN   warehouses wo ON wo.warehouse_id = r.origin_warehouse_id
        JOIN   warehouses wd ON wd.warehouse_id = r.dest_warehouse_id
    """)
    cols   = [d[0] for d in cur.description]
    routes = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return routes


# ── IoT telemetry ─────────────────────────────────────────────────────────────

def generate_iot_telemetry(db, routes):
    print("  Generating IoT telemetry…")
    trucks = [f'TRK-{i:03d}' for i in range(1, 9)]
    batch  = []

    d = date(2023, 1, 1)
    while d <= date(2024, 12, 31):
        if d.weekday() < 5:  # Mon–Fri trips only
            day_routes = random.sample(routes, k=min(len(trucks), len(routes)))
            for truck_id, route in zip(trucks, day_routes):
                sku_ids     = random.sample(list(SKU_TARGET_TEMP.keys()), k=random.randint(1, 3))
                target_temp = min(SKU_TARGET_TEMP[s] for s in sku_ids)
                n_readings  = min(int(route['avg_transit_hours']) + 1, 7)  # max hour = 8+12=20

                for h in range(0, n_readings * 2, 2):   # reading every 2 hours
                    t = h / max(n_readings * 2, 1)
                    lat = route['o_lat'] + (route['d_lat'] - route['o_lat']) * t + random.gauss(0, 0.04)
                    lon = route['o_lon'] + (route['d_lon'] - route['o_lon']) * t + random.gauss(0, 0.04)

                    roll = random.random()
                    if roll < 0.04:                      # 4% high-temp breach
                        temp  = target_temp + random.uniform(3.0, 8.0)
                        alert = 'HIGH_TEMP'
                    elif roll < 0.06:                    # 2% low-temp breach
                        temp  = target_temp - random.uniform(2.0, 5.0)
                        alert = 'LOW_TEMP'
                    else:
                        temp  = target_temp + random.gauss(0.0, 0.5)
                        alert = None

                    batch.append({
                        'truck_id':            truck_id,
                        'timestamp':           datetime.combine(d, dt_time(8 + h, 0)),
                        'route_id':            route['route_id'],
                        'origin_warehouse_id': route['origin_warehouse_id'],
                        'dest_warehouse_id':   route['dest_warehouse_id'],
                        'gps':                 {'lat': round(lat, 5), 'lon': round(lon, 5)},
                        'temperature_c':       round(temp, 2),
                        'humidity_pct':        round(random.uniform(55, 78), 1),
                        'speed_kmh':           round(random.uniform(55, 105), 1),
                        'cargo_sku_ids':       sku_ids,
                        'temp_alert':          alert is not None,
                        'alert_type':          alert,
                    })

                    if len(batch) >= 2000:
                        db.iot_telemetry.insert_many(batch)
                        batch.clear()
        d += timedelta(days=1)

    if batch:
        db.iot_telemetry.insert_many(batch)
    count = db.iot_telemetry.count_documents({})
    print(f"  IoT telemetry: {count:,} documents inserted.")


# ── Weather events ────────────────────────────────────────────────────────────

def generate_weather_events(db):
    print("  Generating weather events…")
    SEVERITY_SCORE = {'low': 1, 'medium': 2, 'high': 3}
    start = datetime(2023, 1, 1)
    events = []

    for _ in range(450):
        ts       = start + timedelta(days=random.randint(0, 729), hours=random.randint(0, 23))
        region   = random.choice(REGIONS)
        season   = 'winter' if ts.month in (11, 12, 1, 2, 3) else 'summer'
        etype    = random.choice(REGION_WEATHER[region][season])
        severity = random.choices(['low', 'medium', 'high'], weights=[0.50, 0.35, 0.15])[0]
        delay    = {
            'low':    random.uniform(0.5, 2.0),
            'medium': random.uniform(2.0, 6.0),
            'high':   random.uniform(6.0, 24.0),
        }[severity]

        events.append({
            'timestamp':            ts,
            'region':               region,
            'event_type':           etype,
            'severity':             severity,
            'severity_score':       SEVERITY_SCORE[severity],
            'expected_delay_hours': round(delay, 1),
            'duration_hours':       round(random.uniform(1, 48), 1),
        })

    db.weather_events.insert_many(events)
    print(f"  Weather events: {len(events)} documents inserted.")


# ── Supplier delays ───────────────────────────────────────────────────────────

def generate_supplier_delays(db):
    print("  Generating supplier delays…")
    start  = datetime(2023, 1, 1)
    delays = []

    for _ in range(200):
        ts       = start + timedelta(days=random.randint(0, 729), hours=random.randint(6, 22))
        supplier = random.choice(SUPPLIERS)
        wh_ids   = random.sample(range(1, 7), k=random.randint(1, 3))
        hrs      = round(random.uniform(2, 72), 1)
        severity = 'low' if hrs < 12 else ('medium' if hrs < 36 else 'high')

        delays.append({
            'timestamp':              ts,
            'supplier_id':            supplier['id'],
            'supplier_name':          supplier['name'],
            'sku_ids':                supplier['sku_ids'],
            'affected_warehouse_ids': wh_ids,
            'delay_hours':            hrs,
            'reason':                 random.choice(DELAY_REASONS),
            'severity':               severity,
        })

    db.supplier_delays.insert_many(delays)
    print(f"  Supplier delays: {len(delays)} documents inserted.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Connecting to MongoDB…")
    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]

    setup_collections(db)

    routes = load_routes()
    print(f"  Loaded {len(routes)} routes from PostgreSQL.")

    generate_iot_telemetry(db, routes)
    generate_weather_events(db)
    generate_supplier_delays(db)

    client.close()
    print("\nMongoDB streaming complete.")


if __name__ == '__main__':
    main()
