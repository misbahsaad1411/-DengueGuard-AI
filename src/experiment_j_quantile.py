
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "dengue_features.csv"
REPORT_DIR = ROOT / "reports"

PREDICTIONS_PATH = REPORT_DIR / "experiment_j_predictions.csv"
SUMMARY_PATH = REPORT_DIR / "experiment_j_summary.json"

TARGET = "target_cases_next_2_4w"

TRAIN_START = "2024-01-01"
VALIDATION_END = "2025-01-01"

HIGH_THRESHOLD = 28.0

QUANTILES = [0.05, 0.10, 0.50, 0.80, 0.90, 0.95]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def create_model(loss="squared_error", quantile=None):

    model_params = {
        "max_iter": 200,
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "random_state": 42,
    }

    if loss == "quantile":
        model_params["loss"] = "quantile"
        model_params["quantile"] = quantile
    else:
        model_params["loss"] = "squared_error"

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    return HistGradientBoostingRegressor(**model_params), categorical_transformer


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
    }


def calculate_high_event_metrics(actual, predicted, threshold):

    actual_high = actual >= threshold
    predicted_high = predicted >= threshold

    return {
        "true_positive": int(np.sum(actual_high & predicted_high)),
        "false_positive": int(np.sum(~actual_high & predicted_high)),
        "false_negative": int(np.sum(actual_high & ~predicted_high)),
        "precision": float(
            precision_score(actual_high, predicted_high, zero_division=0)
        ),
        "recall": float(
            recall_score(actual_high, predicted_high, zero_division=0)
        ),
        "f1": float(
            f1_score(actual_high, predicted_high, zero_division=0)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print("DENGUEGUARD AI - EXPERIMENT J")
    print("QUANTILE REGRESSION AND REPRODUCIBILITY CHECK")
    print("=" * 95)

    # --------------------------------------------------------
    # LOAD DATA - MATCH ORIGINAL VALIDATION SCRIPT
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "date",
            "target_start_date",
            "target_end_date"
        ],
        engine="python"
    )

    # Critical: match evaluate_validation.py row ordering.
    df = df.sort_values(
        ["date", "ward_id"]
    ).reset_index(drop=True)

    excluded_columns = [
        "date",
        "dengue_cases",
        TARGET,
        "target_start_date",
        "target_end_date",
    ]

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    categorical_columns = [
        "city",
        "ward_id"
    ]

    numeric_columns = [
        col for col in feature_columns
        if col not in categorical_columns
    ]

    X = df[feature_columns].copy()
    y = df[TARGET].astype(float)

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    train_mask = (
        (df["date"] < TRAIN_START)
        & (df["target_end_date"] < TRAIN_START)
    )

    validation_mask = (
        (df["date"] >= TRAIN_START)
        & (df["date"] < VALIDATION_END)
        & (df["target_end_date"] < VALIDATION_END)
    )

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_val = X.loc[validation_mask]
    y_val = y.loc[validation_mask]

    print("\nDATA SUMMARY")
    print("-" * 95)
    print(f"Total records: {len(df):,}")
    print(f"Training records: {len(X_train):,}")
    print(f"Validation records: {len(X_val):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Training period: {df.loc[train_mask, 'date'].min().date()} to "
          f"{df.loc[train_mask, 'date'].max().date()}")
    print(f"Validation period: {df.loc[validation_mask, 'date'].min().date()} to "
          f"{df.loc[validation_mask, 'date'].max().date()}")

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            )
        ],
        remainder="passthrough"
    )

    # --------------------------------------------------------
    # TRAIN REFERENCE MODEL A
    # --------------------------------------------------------

    print("\nTRAINING REFERENCE MODEL A")
    print("-" * 95)

    reference_model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42
            )
        )
    ])

    reference_model.fit(X_train, y_train)

    reference_predictions_raw = reference_model.predict(X_val)

    # Match original evaluate_validation.py behavior.
    reference_predictions = np.round(
        np.maximum(reference_predictions_raw, 0),
        2
    )

    actual = y_val.to_numpy()

    reference_metrics = calculate_metrics(
        actual,
        reference_predictions
    )

    reference_high_metrics = calculate_high_event_metrics(
        actual,
        reference_predictions,
        HIGH_THRESHOLD
    )

    print("Reference Model A:")
    print(json.dumps(reference_metrics, indent=2))

    # --------------------------------------------------------
    # TRAIN QUANTILE MODELS
    # --------------------------------------------------------

    predictions = {
        "actual": actual,
        "reference_A": reference_predictions
    }

    quantile_metrics = {}

    for q in QUANTILES:

        print(f"\nTRAINING QUANTILE MODEL: Q={q}")

        model = Pipeline([
            ("preprocessor", ColumnTransformer(
                transformers=[
                    (
                        "categorical",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        ),
                        categorical_columns
                    )
                ],
                remainder="passthrough"
            )),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q,
                    max_iter=200,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42
                )
            )
        ])

        model.fit(X_train, y_train)

        raw_prediction = model.predict(X_val)

        prediction = np.round(
            np.maximum(raw_prediction, 0),
            2
        )

        key = f"q{int(q * 100):02d}"
        predictions[key] = prediction

        metrics = calculate_metrics(actual, prediction)

        metrics["high_event"] = calculate_high_event_metrics(
            actual,
            prediction,
            HIGH_THRESHOLD
        )

        metrics["mean_prediction"] = float(np.mean(prediction))
        metrics["minimum_prediction"] = float(np.min(prediction))
        metrics["maximum_prediction"] = float(np.max(prediction))

        # Quantile coverage: proportion of actual values <= prediction.
        metrics["observed_coverage"] = float(
            np.mean(actual <= prediction)
        )

        quantile_metrics[key] = metrics

        print(f"{key}:")
        print(json.dumps(metrics, indent=2))

    # --------------------------------------------------------
    # PREDICTION INTERVAL DIAGNOSTICS
    # --------------------------------------------------------

    intervals = {}

    interval_pairs = [
        ("q05", "q95"),
        ("q10", "q90"),
    ]

    for lower_key, upper_key in interval_pairs:

        lower = predictions[lower_key]
        upper = predictions[upper_key]

        intervals[f"{lower_key}_{upper_key}"] = {
            "coverage": float(
                np.mean((actual >= lower) & (actual <= upper))
            ),
            "mean_width": float(np.mean(upper - lower)),
            "median_width": float(np.median(upper - lower)),
            "lower_bound_zero_rate": float(np.mean(lower == 0)),
            "crossing_rate": float(np.mean(lower > upper)),
        }

    # --------------------------------------------------------
    # QUANTILE CROSSING
    # --------------------------------------------------------

    crossing = {}

    for lower_key, upper_key in [
        ("q05", "q10"),
        ("q10", "q50"),
        ("q50", "q80"),
        ("q80", "q90"),
        ("q90", "q95"),
    ]:

        crossing[f"{lower_key}_above_{upper_key}"] = float(
            np.mean(
                predictions[lower_key] > predictions[upper_key]
            )
        )

    # --------------------------------------------------------
    # SEVERITY ANALYSIS
    # --------------------------------------------------------

    severity_groups = [
        ("below_28", actual < 28),
        ("28_to_34", (actual >= 28) & (actual < 35)),
        ("35_to_44", (actual >= 35) & (actual < 45)),
        ("45_to_59", (actual >= 45) & (actual < 60)),
        ("60_plus", actual >= 60),
    ]

    severity_results = {}

    for group_name, mask in severity_groups:

        if not np.any(mask):
            severity_results[group_name] = {
                "records": 0
            }
            continue

        severity_results[group_name] = {
            "records": int(np.sum(mask)),
            "reference_A_mae": float(
                mean_absolute_error(
                    actual[mask],
                    predictions["reference_A"][mask]
                )
            ),
            "q50_mae": float(
                mean_absolute_error(
                    actual[mask],
                    predictions["q50"][mask]
                )
            ),
            "q90_coverage": float(
                np.mean(
                    actual[mask] <= predictions["q90"][mask]
                )
            ),
            "q50_underprediction_rate": float(
                np.mean(
                    predictions["q50"][mask] < actual[mask]
                )
            ),
        }

    # --------------------------------------------------------
    # HIGH EVENT COMPARISON
    # --------------------------------------------------------

    high_event_results = {
        "reference_A": reference_high_metrics
    }

    for key in ["q50", "q80", "q90"]:

        high_event_results[key] = calculate_high_event_metrics(
            actual,
            predictions[key],
            HIGH_THRESHOLD
        )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    output = df.loc[
        validation_mask,
        [
            "date",
            "target_start_date",
            "target_end_date",
            "city",
            "ward_id",
            TARGET
        ]
    ].copy()

    output = output.rename(
        columns={TARGET: "actual"}
    )

    for key in predictions:

        if key == "actual":
            continue

        output[key] = predictions[key]

    output.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary = {
        "experiment": "J - Quantile Regression",
        "training_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "feature_count": int(len(feature_columns)),
        "high_event_threshold": HIGH_THRESHOLD,
        "reference_A": {
            "metrics": reference_metrics,
            "high_event": reference_high_metrics
        },
        "quantile_models": quantile_metrics,
        "prediction_intervals": intervals,
        "quantile_crossing": crossing,
        "severity_analysis": severity_results,
        "high_event_comparison": high_event_results,
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 95)
    print("EXPERIMENT J FINAL SUMMARY")
    print("=" * 95)

    print("\nREFERENCE MODEL A")
    print(json.dumps(reference_metrics, indent=2))

    print("\nQUANTILE MODEL MAE COMPARISON")

    print(f"{'Model':<15}{'MAE':>12}{'RMSE':>12}{'R2':>12}{'Bias':>12}")

    print(
        f"{'Reference A':<15}"
        f"{reference_metrics['mae']:>12.4f}"
        f"{reference_metrics['rmse']:>12.4f}"
        f"{reference_metrics['r2']:>12.4f}"
        f"{reference_metrics['bias']:>12.4f}"
    )

    for key, metrics in quantile_metrics.items():

        print(
            f"{key:<15}"
            f"{metrics['mae']:>12.4f}"
            f"{metrics['rmse']:>12.4f}"
            f"{metrics['r2']:>12.4f}"
            f"{metrics['bias']:>12.4f}"
        )

    print("\nPREDICTION INTERVAL DIAGNOSTICS")
    print(json.dumps(intervals, indent=2))

    print("\nQUANTILE CROSSING")
    print(json.dumps(crossing, indent=2))

    print("\nSEVERITY ANALYSIS")
    print(json.dumps(severity_results, indent=2))

    print("\nHIGH EVENT COMPARISON")
    print(json.dumps(high_event_results, indent=2))

    print("\nSAVED FILES")
    print(PREDICTIONS_PATH)
    print(SUMMARY_PATH)

    print("\nEXPERIMENT J COMPLETE")


if __name__ == "__main__":
    main()
