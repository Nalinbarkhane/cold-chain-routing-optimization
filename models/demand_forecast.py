import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.csv')
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'demand_forecast.json')
ENCODE_PATH = os.path.join(os.path.dirname(__file__), 'label_encoders.pkl')
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), 'plots')

TEST_START = '2024-10-01'   # last 3 months held out as test
VAL_START  = '2024-08-01'   # prior 2 months used for early-stopping validation

FEATURES = [
    # Temporal
    'day_of_week', 'month', 'quarter', 'week_of_year', 'day_of_year', 'year', 'is_weekend',
    # Entity IDs
    'sku_id', 'warehouse_id', 'category_enc', 'region_enc',
    # Static SKU attributes
    'temp_min_c', 'temp_max_c', 'unit_cost', 'shelf_life_days',
    # Lag features
    'lag_7', 'lag_14', 'lag_28',
    # Rolling statistics
    'rolling_7_mean', 'rolling_14_mean', 'rolling_28_mean', 'rolling_7_std',
    # External signals
    'max_severity_score', 'total_delay_hours', 'had_weather_event',
    'breach_count', 'supplier_delay_hrs', 'had_supplier_delay',
]
TARGET = 'demand_units'


# ── Data preparation ──────────────────────────────────────────────────────────

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])

    # Drop early rows where lag windows are incomplete
    df = df.dropna(subset=['lag_7', 'lag_14', 'lag_28']).reset_index(drop=True)

    # Encode categorical columns as integers
    label_encoders = {}
    for col in ['category', 'region']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col])
        label_encoders[col] = le

    with open(ENCODE_PATH, 'wb') as f:
        pickle.dump(label_encoders, f)

    return df, label_encoders


def split_data(df: pd.DataFrame):
    train_mask = df['date'] <  VAL_START
    val_mask   = (df['date'] >= VAL_START) & (df['date'] < TEST_START)
    test_mask  = df['date'] >= TEST_START

    X_tr   = df.loc[train_mask, FEATURES]
    y_tr   = df.loc[train_mask, TARGET]
    X_val  = df.loc[val_mask,   FEATURES]
    y_val  = df.loc[val_mask,   TARGET]
    X_test = df.loc[test_mask,  FEATURES]
    y_test = df.loc[test_mask,  TARGET]

    print(f"  Train : {train_mask.sum():>7,} rows  ({df.loc[train_mask,'date'].min().date()} → {df.loc[train_mask,'date'].max().date()})")
    print(f"  Val   : {val_mask.sum():>7,} rows  ({df.loc[val_mask,'date'].min().date()} → {df.loc[val_mask,'date'].max().date()})")
    print(f"  Test  : {test_mask.sum():>7,} rows  ({df.loc[test_mask,'date'].min().date()} → {df.loc[test_mask,'date'].max().date()})")

    return X_tr, y_tr, X_val, y_val, X_test, y_test


# ── Training ──────────────────────────────────────────────────────────────────

def train(X_tr, y_tr, X_val, y_val) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators        = 1000,
        max_depth           = 6,
        learning_rate       = 0.05,
        subsample           = 0.80,
        colsample_bytree    = 0.80,
        min_child_weight    = 3,
        gamma               = 0.10,
        reg_alpha           = 0.10,
        reg_lambda          = 1.00,
        objective           = 'reg:squarederror',
        eval_metric         = 'rmse',
        early_stopping_rounds = 50,
        random_state        = 42,
        n_jobs              = -1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    print(f"\n  Best iteration: {model.best_iteration}  |  Val RMSE: {model.best_score:.2f}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, df_test: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(X_test).clip(min=0)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test.values - preds) / np.maximum(y_test.values, 1))) * 100
    r2   = r2_score(y_test, preds)

    print("\n── Test Set Metrics ───────────────────────────────")
    print(f"  MAE   : {mae:.2f} units")
    print(f"  RMSE  : {rmse:.2f} units")
    print(f"  MAPE  : {mape:.2f} %")
    print(f"  R²    : {r2:.4f}")
    print("───────────────────────────────────────────────────")

    results = df_test.copy()
    results['predicted'] = preds
    results['error']     = results[TARGET] - results['predicted']
    results['abs_error'] = results['error'].abs()
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_feature_importance(model):
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(9, 7))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('XGBoost Feature Importance (Gain) — Top 20', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.axvline(importances.mean(), color='firebrick', linestyle='--', linewidth=1, label='Mean')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150)
    plt.close(fig)
    print("  Saved: feature_importance.png")


def plot_actual_vs_predicted(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 8))
    palette = sns.color_palette('tab10', n_colors=results['category'].nunique())
    for i, (cat, grp) in enumerate(results.groupby('category')):
        ax.scatter(grp[TARGET], grp['predicted'], alpha=0.25, s=5,
                   color=palette[i], label=cat)
    lim = max(results[TARGET].max(), results['predicted'].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1.2, label='Perfect prediction')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('Actual Demand (units)');  ax.set_ylabel('Predicted Demand (units)')
    ax.set_title('Actual vs. Predicted — Test Set (Oct–Dec 2024)', fontsize=13, fontweight='bold')
    ax.legend(markerscale=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'actual_vs_predicted.png'), dpi=150)
    plt.close(fig)
    print("  Saved: actual_vs_predicted.png")


def plot_time_series(df: pd.DataFrame, model, label_encoders):
    # Four representative (sku_id, warehouse_id) combos with distinct seasonality
    combos = [
        (2,  3, 'Flu Vaccines — Chicago (Winter Peak)'),
        (11, 5, 'Premium Ice Cream — LA (Summer Peak)'),
        (8,  1, 'Leafy Greens — Boston (Produce Season)'),
        (5,  3, 'Whole Milk — Chicago (Stable / High Volume)'),
    ]

    # Prepare feature matrix for the full series of selected combos
    context_start = '2024-07-01'
    df_ctx = df[df['date'] >= context_start].copy()

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)
    fig.suptitle('Demand Forecast: Actual vs. Predicted (Jul–Dec 2024)',
                 fontsize=14, fontweight='bold', y=1.01)

    for ax, (sku_id, wh_id, title) in zip(axes, combos):
        mask   = (df_ctx['sku_id'] == sku_id) & (df_ctx['warehouse_id'] == wh_id)
        subset = df_ctx[mask].copy()
        if subset.empty:
            ax.set_title(title); ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes)
            continue

        preds = model.predict(subset[FEATURES]).clip(min=0)
        ax.plot(subset['date'], subset[TARGET],   color='steelblue',  linewidth=1.4, label='Actual')
        ax.plot(subset['date'], preds,            color='tomato',     linewidth=1.4, label='Predicted', linestyle='--')
        ax.axvline(pd.to_datetime(TEST_START), color='gray', linestyle=':', linewidth=1.2, label='Test start')
        ax.axvspan(pd.to_datetime(TEST_START), subset['date'].max(),
                   alpha=0.07, color='tomato', label='Test window')
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Units'); ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'time_series_forecast.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: time_series_forecast.png")


def plot_error_distribution(results: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Error histogram
    axes[0].hist(results['error'], bins=80, color='steelblue', edgecolor='white', linewidth=0.3)
    axes[0].axvline(0, color='firebrick', linestyle='--', linewidth=1.5)
    axes[0].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Actual − Predicted (units)'); axes[0].set_ylabel('Frequency')

    # MAPE by category
    cat_mape = (
        results.groupby('category')
        .apply(lambda g: np.mean(np.abs(g['error']) / np.maximum(g[TARGET], 1)) * 100,
               include_groups=False)
        .sort_values()
    )
    cat_mape.plot(kind='barh', ax=axes[1], color='steelblue')
    axes[1].set_title('MAPE by Product Category — Test Set', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('MAPE (%)')
    for bar, val in zip(axes[1].patches, cat_mape):
        axes[1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                     f'{val:.1f}%', va='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'error_distribution.png'), dpi=150)
    plt.close(fig)
    print("  Saved: error_distribution.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading and preparing data…")
    df, label_encoders = load_and_prepare(DATA_PATH)
    print(f"  {len(df):,} rows after dropping incomplete lag windows.\n")

    print("Splitting data…")
    X_tr, y_tr, X_val, y_val, X_test, y_test = split_data(df)

    print("\nTraining XGBoost…")
    model = train(X_tr, y_tr, X_val, y_val)

    print("\nEvaluating on test set…")
    test_df  = df[df['date'] >= TEST_START].reset_index(drop=True)
    results  = evaluate(model, X_test, y_test, test_df)

    print("\nGenerating plots…")
    plot_feature_importance(model)
    plot_actual_vs_predicted(results)
    plot_time_series(df, model, label_encoders)
    plot_error_distribution(results)

    model.save_model(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Plots saved → {PLOTS_DIR}/")
    print("\nDemand forecasting complete.")


if __name__ == '__main__':
    main()
