import os
import psycopg2
import pandas as pd
from pymongo import MongoClient

PG_CONFIG  = {'host': 'localhost', 'port': 5432, 'dbname': 'cold_chain', 'user': 'nalin'}
MONGO_URI  = 'mongodb://localhost:27017/'
MONGO_DB   = 'cold_chain'
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.csv')


# ── Extract ───────────────────────────────────────────────────────────────────

def load_demand_sql(conn) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT
            d.date,
            d.sku_id,
            d.warehouse_id,
            d.demand_units,
            d.fulfilled_units,
            d.stockout_flag,
            s.name            AS sku_name,
            s.category,
            s.temp_min_c,
            s.temp_max_c,
            s.unit_cost,
            s.shelf_life_days,
            w.name            AS warehouse_name,
            w.region
        FROM daily_demand   d
        JOIN skus           s ON s.sku_id       = d.sku_id
        JOIN warehouses     w ON w.warehouse_id = d.warehouse_id
        ORDER BY d.date, d.sku_id, d.warehouse_id
    """, conn, parse_dates=['date'])


def load_weather_mongo(db) -> pd.DataFrame:
    """Aggregate weather events to (date, region) granularity."""
    pipeline = [
        {'$group': {
            '_id': {
                'date':   {'$dateToString': {'format': '%Y-%m-%d', 'date': '$timestamp'}},
                'region': '$region',
            },
            'max_severity_score': {'$max':  '$severity_score'},
            'total_delay_hours':  {'$sum':  '$expected_delay_hours'},
            'event_count':        {'$sum':  1},
        }},
        {'$project': {
            '_id':                0,
            'date':               '$_id.date',
            'region':             '$_id.region',
            'max_severity_score': 1,
            'total_delay_hours':  1,
            'event_count':        1,
        }},
    ]
    docs = list(db.weather_events.aggregate(pipeline))
    if not docs:
        return pd.DataFrame(columns=['date', 'region', 'max_severity_score',
                                     'total_delay_hours', 'event_count'])
    df = pd.DataFrame(docs)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_iot_mongo(db) -> pd.DataFrame:
    """Count temperature breaches per destination warehouse per date."""
    pipeline = [
        {'$match': {'temp_alert': True}},
        {'$group': {
            '_id': {
                'date':         {'$dateToString': {'format': '%Y-%m-%d', 'date': '$timestamp'}},
                'warehouse_id': '$dest_warehouse_id',
            },
            'breach_count': {'$sum': 1},
        }},
        {'$project': {
            '_id':          0,
            'date':         '$_id.date',
            'warehouse_id': '$_id.warehouse_id',
            'breach_count': 1,
        }},
    ]
    docs = list(db.iot_telemetry.aggregate(pipeline))
    if not docs:
        return pd.DataFrame(columns=['date', 'warehouse_id', 'breach_count'])
    df = pd.DataFrame(docs)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_supplier_delays_mongo(db) -> pd.DataFrame:
    """Expand supplier delay documents to (date, warehouse_id, sku_id) rows."""
    docs = list(db.supplier_delays.find(
        {},
        {'_id': 0, 'timestamp': 1, 'sku_ids': 1, 'affected_warehouse_ids': 1, 'delay_hours': 1},
    ))
    if not docs:
        return pd.DataFrame(columns=['date', 'warehouse_id', 'sku_id', 'supplier_delay_hrs'])

    rows = [
        {'date': doc['timestamp'].date(), 'warehouse_id': wh, 'sku_id': sku,
         'delay_hours': doc['delay_hours']}
        for doc in docs
        for wh  in doc['affected_warehouse_ids']
        for sku in doc['sku_ids']
    ]
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = (df.groupby(['date', 'warehouse_id', 'sku_id'])['delay_hours']
            .sum().reset_index()
            .rename(columns={'delay_hours': 'supplier_delay_hrs'}))
    return df


# ── Transform ─────────────────────────────────────────────────────────────────

def engineer_features(demand: pd.DataFrame,
                       weather: pd.DataFrame,
                       iot: pd.DataFrame,
                       supplier: pd.DataFrame) -> pd.DataFrame:

    df = demand.sort_values(['sku_id', 'warehouse_id', 'date']).reset_index(drop=True)

    # ── Time features
    df['day_of_week']  = df['date'].dt.dayofweek
    df['month']        = df['date'].dt.month
    df['quarter']      = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_year']  = df['date'].dt.dayofyear
    df['year']         = df['date'].dt.year
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)

    # ── Lag features (within each sku × warehouse series)
    for lag in [7, 14, 28]:
        df[f'lag_{lag}'] = df.groupby(['sku_id', 'warehouse_id'])['demand_units'].shift(lag)

    # ── Rolling window features (shift 1 day to prevent leakage)
    for window in [7, 14, 28]:
        df[f'rolling_{window}_mean'] = (
            df.groupby(['sku_id', 'warehouse_id'])['demand_units']
            .transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean())
        )
    df['rolling_7_std'] = (
        df.groupby(['sku_id', 'warehouse_id'])['demand_units']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std().fillna(0))
    )

    # ── Join weather by (date, region)
    df = df.merge(weather, on=['date', 'region'], how='left')
    df['max_severity_score'] = df['max_severity_score'].fillna(0).astype(int)
    df['total_delay_hours']  = df['total_delay_hours'].fillna(0)
    df['had_weather_event']  = (df['event_count'].fillna(0) > 0).astype(int)
    df.drop(columns=['event_count'], inplace=True)

    # ── Join IoT breaches — shift forward 1 day (yesterday's breach → today's risk)
    iot_lagged       = iot.copy()
    iot_lagged['date'] = iot_lagged['date'] + pd.Timedelta(days=1)
    df = df.merge(iot_lagged, on=['date', 'warehouse_id'], how='left')
    df['breach_count'] = df['breach_count'].fillna(0).astype(int)

    # ── Join supplier delays by (date, warehouse_id, sku_id)
    df = df.merge(supplier, on=['date', 'warehouse_id', 'sku_id'], how='left')
    df['supplier_delay_hrs'] = df['supplier_delay_hrs'].fillna(0)
    df['had_supplier_delay'] = (df['supplier_delay_hrs'] > 0).astype(int)

    df.reset_index(drop=True, inplace=True)
    return df


# ── Load ──────────────────────────────────────────────────────────────────────

def main():
    print("Starting ETL pipeline…")

    conn   = psycopg2.connect(**PG_CONFIG)
    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]

    print("  [1/4] Loading demand from PostgreSQL…")
    demand = load_demand_sql(conn)
    print(f"        → {len(demand):,} rows")

    print("  [2/4] Loading weather aggregations from MongoDB…")
    weather = load_weather_mongo(db)
    print(f"        → {len(weather):,} (date × region) aggregations")

    print("  [3/4] Loading IoT breach aggregations from MongoDB…")
    iot = load_iot_mongo(db)
    print(f"        → {len(iot):,} (date × warehouse) aggregations")

    print("  [4/4] Loading supplier delays from MongoDB…")
    supplier = load_supplier_delays_mongo(db)
    print(f"        → {len(supplier):,} (date × warehouse × sku) rows")

    print("\n  Engineering features…")
    features = engineer_features(demand, weather, iot, supplier)
    print(f"  Feature matrix: {features.shape[0]:,} rows × {features.shape[1]} columns")
    print(f"  Columns: {list(features.columns)}")

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    features.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved → {os.path.abspath(OUTPUT_CSV)}")

    conn.close()
    client.close()
    print("\nETL complete.")


if __name__ == '__main__':
    main()
