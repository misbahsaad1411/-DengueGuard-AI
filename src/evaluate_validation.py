
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "dengue_features.csv"

PREDICTIONS_PATH = ROOT / "reports" / "validation_predictions.csv"
REPORT_PATH = ROOT / "reports" / "validation_diagnostics.json"

TARGET = "target_cases_next_2_4w"

VALIDATION_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")

CATEGORICAL_FEATURES = ["city", "ward_id"]

EXCLUDED_COLUMNS = [
    "date",
    "dengue_cases",
    TARGET,
    "target_start_date",
    "target_end_date"
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float),
        0
    )

    if len(actual) == 0:
        return {}

    return {
        "records": int(len(actual)),
        "actual_mean": round(float(np.mean(actual)), 4),
        "predicted_mean": round(float(np.mean(predicted)), 4),
        "mae": round(
            float(mean_absolute_error(actual, predicted)), 4
        ),
        "rmse": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 4
        ),
        "r2": round(float(r2_score(actual, predicted)), 4),
        "bias": round(float(np.mean(predicted - actual)), 4)
    }


def grouped_metrics(df, group_column):

    results = {}

    for group_name, group in df.groupby(group_column):

        results[str(group_name)] = calculate_metrics(
            group["actual_cases"],
            group["predicted_cases"]
        )

    return results


# ============================================================
# MODEL PIPELINE
# ============================================================

def create_pipeline(numeric_features):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                "passthrough",
                numeric_features
            )
        ],
        remainder="drop"
    )

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DENGUEGUARD AI - VALIDATION DIAGNOSTICS")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "date",
            "target_start_date",
            "target_end_date"
        ]
    )

    df = df.sort_values(
        ["date", "ward_id"]
    ).reset_index(drop=True)

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    numeric_features = [
        column
        for column in feature_columns
        if column not in CATEGORICAL_FEATURES
    ]

    if df[feature_columns].isnull().any().any():
        raise ValueError("Missing values found in model features.")

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    train_mask = (
        (df["date"] < VALIDATION_START)
        & (df["target_end_date"] < VALIDATION_START)
    )

    validation_mask = (
        (df["date"] >= VALIDATION_START)
        & (df["date"] < TEST_START)
        & (df["target_end_date"] < TEST_START)
    )

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()

    if train_df.empty or validation_df.empty:
        raise ValueError("Training or validation dataset is empty.")

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df[TARGET]

    print("\nDATA SPLIT")
    print("-" * 70)
    print("Training records:", len(train_df))
    print("Validation records:", len(validation_df))
    print("Training period:",
          train_df["date"].min().date(),
          "to",
          train_df["date"].max().date())
    print("Validation period:",
          validation_df["date"].min().date(),
          "to",
          validation_df["date"].max().date())

    # --------------------------------------------------------
    # TRAINING-DATA HIGH-INCIDENCE THRESHOLD
    # --------------------------------------------------------

    high_threshold = float(y_train.quantile(0.90))

    print("\nTRAINING-DATA HIGH-INCIDENCE THRESHOLD")
    print("-" * 70)
    print("90th percentile:", round(high_threshold, 4))

    # --------------------------------------------------------
    # TRAIN EXPERIMENT A
    # --------------------------------------------------------

    print("\nTraining Experiment A: HistGradientBoosting...")

    pipeline = create_pipeline(numeric_features)

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = np.maximum(
        pipeline.predict(X_validation),
        0
    )

    # Existing baseline from the original training script
    baseline_predictions = (
        validation_df["dengue_cases_rolling_4w"].to_numpy() * 3
    )

    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
    # --------------------------------------------------------

    results_df = pd.DataFrame({
        "date": validation_df["date"].values,
        "target_start_date": validation_df["target_start_date"].values,
        "target_end_date": validation_df["target_end_date"].values,
        "city": validation_df["city"].values,
        "ward_id": validation_df["ward_id"].values,
        "actual_cases": y_validation.values,
        "predicted_cases": np.round(predictions, 2),
        "baseline_predictions": np.round(
            baseline_predictions, 2
        )
    })

    results_df["absolute_error"] = (
        results_df["predicted_cases"]
        - results_df["actual_cases"]
    ).abs()

    results_df["signed_error"] = (
        results_df["predicted_cases"]
        - results_df["actual_cases"]
    )

    results_df["month"] = (
        pd.to_datetime(results_df["date"]).dt.month
    )

    results_df["high_incidence_actual"] = (
        results_df["actual_cases"] >= high_threshold
    )

    results_df["high_incidence_predicted"] = (
        results_df["predicted_cases"] >= high_threshold
    )

    # --------------------------------------------------------
    # HIGH-INCIDENCE ANALYSIS
    # --------------------------------------------------------

    high_df = results_df[
        results_df["high_incidence_actual"]
    ]

    high_incidence_metrics = calculate_metrics(
        high_df["actual_cases"],
        high_df["predicted_cases"]
    )

    if len(high_df):

        actual_high_underprediction_rate = float(
            (
                high_df["predicted_cases"]
                < high_df["actual_cases"]
            ).mean()
        )

        high_event_missed_rate = float(
            (
                high_df["predicted_cases"]
                < high_threshold
            ).mean()
        )

    else:
        actual_high_underprediction_rate = None
        high_event_missed_rate = None

    # --------------------------------------------------------
    # OVERALL AND GROUPED METRICS
    # --------------------------------------------------------

    overall = calculate_metrics(
        results_df["actual_cases"],
        results_df["predicted_cases"]
    )

    baseline = calculate_metrics(
        results_df["actual_cases"],
        results_df["baseline_predictions"]
    )

    city_performance = grouped_metrics(
        results_df,
        "city"
    )

    monthly_performance = grouped_metrics(
        results_df,
        "month"
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {
        "experiment": "A - Existing HistGradientBoosting",
        "target": TARGET,
        "target_definition": (
            "Sum of dengue cases at weeks t+2, t+3 and t+4"
        ),
        "training_records": int(len(train_df)),
        "validation_records": int(len(validation_df)),
        "training_start": str(train_df["date"].min().date()),
        "training_end": str(train_df["date"].max().date()),
        "validation_start": str(validation_df["date"].min().date()),
        "validation_end": str(validation_df["date"].max().date()),
        "feature_count": int(len(feature_columns)),
        "high_incidence_threshold_from_training": round(
            high_threshold, 4
        ),
        "overall_performance": overall,
        "baseline_performance": baseline,
        "city_performance": city_performance,
        "monthly_performance_by_prediction_date": monthly_performance,
        "high_incidence_performance": {
            **high_incidence_metrics,
            "actual_high_underprediction_rate": (
                round(actual_high_underprediction_rate, 4)
                if actual_high_underprediction_rate is not None
                else None
            ),
            "high_event_missed_rate": (
                round(high_event_missed_rate, 4)
                if high_event_missed_rate is not None
                else None
            )
        },
        "underprediction_records": int(
            (results_df["signed_error"] < 0).sum()
        ),
        "overprediction_records": int(
            (results_df["signed_error"] > 0).sum()
        ),
        "feature_columns": feature_columns
    }

    # --------------------------------------------------------
    # SAVE WITHOUT OVERWRITING EXISTING MODEL OR TEST REPORTS
    # --------------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\nOVERALL VALIDATION PERFORMANCE")
    print("-" * 70)
    print(json.dumps(overall, indent=4))

    print("\nBASELINE PERFORMANCE")
    print("-" * 70)
    print(json.dumps(baseline, indent=4))

    print("\nCITY PERFORMANCE")
    print(json.dumps(city_performance, indent=4))

    print("\nMONTHLY PERFORMANCE")
    print(json.dumps(monthly_performance, indent=4))

    print("\nHIGH-INCIDENCE PERFORMANCE")
    print(json.dumps(report["high_incidence_performance"], indent=4))

    print("\nUNDERPREDICTION RECORDS:",
          report["underprediction_records"])

    print("OVERPREDICTION RECORDS:",
          report["overprediction_records"])

    print("\nSAVED FILES")
    print("-" * 70)
    print("Validation predictions:", PREDICTIONS_PATH)
    print("Validation diagnostics:", REPORT_PATH)

    print("\nEXPERIMENT A VALIDATION COMPLETED.")


if __name__ == "__main__":
    main()
