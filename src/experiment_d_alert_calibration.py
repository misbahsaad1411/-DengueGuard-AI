
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# DENGUEGUARD AI - EXPERIMENT D
# ALERT THRESHOLD CALIBRATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

DATA_PATH = ROOT / "data" / "processed" / "dengue_features.csv"
VALIDATION_PATH = REPORTS / "validation_predictions.csv"

OUTPUT_2023 = REPORTS / "experiment_d_2023_calibration.csv"
OUTPUT_2024 = REPORTS / "experiment_d_2024_evaluation.csv"
OUTPUT_JSON = REPORTS / "experiment_d_diagnostics.json"

TARGET = "target_cases_next_2_4w"

TRAIN_START = pd.Timestamp("2023-01-01")
CALIBRATION_END = pd.Timestamp("2024-01-01")
VALIDATION_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")

# Keep the existing high-incidence outcome definition.
# This is the actual-case threshold, NOT the prediction threshold.
ACTUAL_HIGH_THRESHOLD = 28.0

# Candidate prediction alert thresholds.
# Threshold is selected using 2023 only.
CANDIDATE_THRESHOLDS = np.arange(8.0, 40.01, 0.5)

CATEGORICAL_FEATURES = ["city", "ward_id"]

EXCLUDED_COLUMNS = [
    "date",
    "dengue_cases",
    TARGET,
    "target_start_date",
    "target_end_date",
]


# ============================================================
# MODEL
# Same HistGradientBoosting configuration as Experiment A.
# ============================================================

def build_model(numeric_features):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                "passthrough",
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    regressor = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def evaluate_alerts(actual, predicted, alert_threshold):

    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float),
        0,
    )

    actual_high = actual >= ACTUAL_HIGH_THRESHOLD
    predicted_high = predicted >= alert_threshold

    precision = precision_score(
        actual_high,
        predicted_high,
        zero_division=0,
    )

    recall = recall_score(
        actual_high,
        predicted_high,
        zero_division=0,
    )

    f1 = f1_score(
        actual_high,
        predicted_high,
        zero_division=0,
    )

    tp = int(np.sum(actual_high & predicted_high))
    fp = int(np.sum(~actual_high & predicted_high))
    fn = int(np.sum(actual_high & ~predicted_high))
    tn = int(np.sum(~actual_high & ~predicted_high))

    return {
        "prediction_alert_threshold": round(float(alert_threshold), 4),
        "actual_high_threshold": ACTUAL_HIGH_THRESHOLD,
        "records": int(len(actual)),
        "actual_high_events": int(np.sum(actual_high)),
        "alerts_issued": int(np.sum(predicted_high)),
        "true_positive": tp,
        "false_positive": fp,
        "missed_events": fn,
        "true_negative": tn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "miss_rate": round(float(1 - recall), 4),
        "mae": round(
            float(mean_absolute_error(actual, predicted)),
            4,
        ),
    }


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

def select_alert_threshold(actual, predicted):

    results = []

    for threshold in CANDIDATE_THRESHOLDS:

        metrics = evaluate_alerts(
            actual,
            predicted,
            threshold,
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Select maximum F1 on 2023 calibration data.
    # If tied, prefer higher precision.
    # If still tied, prefer the higher alert threshold.
    results_df = results_df.sort_values(
        ["f1", "precision", "prediction_alert_threshold"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best = results_df.iloc[0].to_dict()

    return float(best["prediction_alert_threshold"]), results_df


# ============================================================
# MAIN
# ============================================================

def main():

    REPORTS.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("DENGUEGUARD AI - EXPERIMENT D")
    print("ALERT THRESHOLD CALIBRATION")
    print("=" * 85)

    # --------------------------------------------------------
    # 1. LOAD FEATURE DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "date",
            "target_start_date",
            "target_end_date",
        ],
    )

    df = df.sort_values(
        ["date", "ward_id"]
    ).reset_index(drop=True)

    feature_columns = [
        col for col in df.columns
        if col not in EXCLUDED_COLUMNS
    ]

    missing_required = [
        col for col in CATEGORICAL_FEATURES + EXCLUDED_COLUMNS
        if col not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}"
        )

    numeric_features = [
        col for col in feature_columns
        if col not in CATEGORICAL_FEATURES
    ]

    if df[feature_columns].isnull().any().any():
        raise ValueError(
            "Missing values found in model features."
        )

    print("\nFEATURES")
    print("-" * 85)
    print("Feature count:", len(feature_columns))

    # --------------------------------------------------------
    # 2. TRAIN MODEL USING DATA BEFORE 2023
    # --------------------------------------------------------

    train_mask = (
        (df["date"] < TRAIN_START)
        & (df["target_end_date"] < TRAIN_START)
    )

    calibration_mask = (
        (df["date"] >= TRAIN_START)
        & (df["date"] < CALIBRATION_END)
        & (df["target_end_date"] < CALIBRATION_END)
    )

    train_df = df.loc[train_mask].copy()
    calibration_df = df.loc[calibration_mask].copy()

    if train_df.empty or calibration_df.empty:
        raise ValueError(
            "Training or 2023 calibration data is empty."
        )

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]

    X_calibration = calibration_df[feature_columns]
    y_calibration = calibration_df[TARGET]

    print("\nTEMPORAL SPLIT")
    print("-" * 85)
    print("Training records:", len(train_df))
    print("Calibration records:", len(calibration_df))

    print(
        "Training dates:",
        train_df["date"].min().date(),
        "to",
        train_df["date"].max().date(),
    )

    print(
        "Calibration dates:",
        calibration_df["date"].min().date(),
        "to",
        calibration_df["date"].max().date(),
    )

    print("\nTRAINING CALIBRATION MODEL...")

    calibration_model = build_model(numeric_features)

    calibration_model.fit(
        X_train,
        y_train,
    )

    calibration_predictions = np.maximum(
        calibration_model.predict(X_calibration),
        0,
    )

    # --------------------------------------------------------
    # 3. CALIBRATE ALERT THRESHOLD ON 2023
    # --------------------------------------------------------

    print("\nCALIBRATING ALERT THRESHOLD USING 2023...")
    print("-" * 85)

    calibrated_threshold, threshold_results = (
        select_alert_threshold(
            y_calibration,
            calibration_predictions,
        )
    )

    calibration_at_28 = evaluate_alerts(
        y_calibration,
        calibration_predictions,
        ACTUAL_HIGH_THRESHOLD,
    )

    calibration_at_selected = evaluate_alerts(
        y_calibration,
        calibration_predictions,
        calibrated_threshold,
    )

    print("Actual high-incidence threshold:", ACTUAL_HIGH_THRESHOLD)
    print("Selected prediction alert threshold:", calibrated_threshold)

    print("\n2023 CALIBRATION COMPARISON")
    print("-" * 85)

    print(
        f"{'Metric':<25}"
        f"{'Threshold 28':<20}"
        f"{'Calibrated':<20}"
    )

    for metric in [
        "alerts_issued",
        "true_positive",
        "false_positive",
        "missed_events",
        "precision",
        "recall",
        "f1",
        "miss_rate",
    ]:

        print(
            f"{metric:<25}"
            f"{str(calibration_at_28[metric]):<20}"
            f"{str(calibration_at_selected[metric]):<20}"
        )

    # Save 2023 predictions and selected threshold.
    calibration_output = calibration_df[
        [
            "date",
            "target_start_date",
            "target_end_date",
            "city",
            "ward_id",
        ]
    ].copy()

    calibration_output["actual_cases"] = y_calibration.to_numpy()
    calibration_output["predicted_cases"] = calibration_predictions
    calibration_output["alert_at_28"] = (
        calibration_predictions >= ACTUAL_HIGH_THRESHOLD
    )
    calibration_output["alert_at_calibrated_threshold"] = (
        calibration_predictions >= calibrated_threshold
    )

    calibration_output.to_csv(
        OUTPUT_2023,
        index=False,
    )

    # --------------------------------------------------------
    # 4. LOAD EXISTING 2024 EXPERIMENT A PREDICTIONS
    # --------------------------------------------------------

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Experiment A validation predictions not found: {VALIDATION_PATH}"
        )

    validation = pd.read_csv(
        VALIDATION_PATH,
        parse_dates=[
            "date",
            "target_start_date",
            "target_end_date",
        ],
    )

    required_validation_columns = [
        "date",
        "target_start_date",
        "target_end_date",
        "city",
        "ward_id",
        "actual_cases",
        "predicted_cases",
    ]

    missing = [
        col for col in required_validation_columns
        if col not in validation.columns
    ]

    if missing:
        raise ValueError(
            f"Missing validation columns: {missing}"
        )

    validation = validation.sort_values(
        ["date", "ward_id"]
    ).reset_index(drop=True)

    if not (
        (validation["date"] >= VALIDATION_START)
        & (validation["date"] < TEST_START)
        & (validation["target_end_date"] < TEST_START)
    ).all():
        raise ValueError(
            "Validation file contains records outside the expected 2024 period."
        )

    actual_2024 = validation["actual_cases"].to_numpy(dtype=float)
    predicted_2024 = np.maximum(
        validation["predicted_cases"].to_numpy(dtype=float),
        0,
    )

    # --------------------------------------------------------
    # 5. EVALUATE CALIBRATED THRESHOLD ON 2024
    # --------------------------------------------------------

    evaluation_at_28 = evaluate_alerts(
        actual_2024,
        predicted_2024,
        ACTUAL_HIGH_THRESHOLD,
    )

    evaluation_at_selected = evaluate_alerts(
        actual_2024,
        predicted_2024,
        calibrated_threshold,
    )

    print("\n" + "=" * 85)
    print("2024 OUT-OF-SAMPLE EVALUATION")
    print("=" * 85)

    print(
        f"{'Metric':<25}"
        f"{'Threshold 28':<20}"
        f"{'Calibrated':<20}"
    )

    for metric in [
        "alerts_issued",
        "true_positive",
        "false_positive",
        "missed_events",
        "precision",
        "recall",
        "f1",
        "miss_rate",
        "mae",
    ]:

        print(
            f"{metric:<25}"
            f"{str(evaluation_at_28[metric]):<20}"
            f"{str(evaluation_at_selected[metric]):<20}"
        )

    # --------------------------------------------------------
    # 6. MONTHLY 2024 ANALYSIS
    # --------------------------------------------------------

    midpoint = (
        validation["target_start_date"]
        + (
            validation["target_end_date"]
            - validation["target_start_date"]
        ) / 2
    )

    validation["target_month"] = midpoint.dt.month

    monthly_results = []

    for month, group in validation.groupby("target_month"):

        actual = group["actual_cases"].to_numpy(dtype=float)
        predicted = np.maximum(
            group["predicted_cases"].to_numpy(dtype=float),
            0,
        )

        baseline_metrics = evaluate_alerts(
            actual,
            predicted,
            ACTUAL_HIGH_THRESHOLD,
        )

        calibrated_metrics = evaluate_alerts(
            actual,
            predicted,
            calibrated_threshold,
        )

        month_name = pd.Timestamp(
            year=2024,
            month=int(month),
            day=1,
        ).strftime("%B")

        monthly_results.append({
            "target_month": month_name,
            "records": len(group),
            "actual_high_events": baseline_metrics["actual_high_events"],
            "alerts_at_28": baseline_metrics["alerts_issued"],
            "recall_at_28": baseline_metrics["recall"],
            "f1_at_28": baseline_metrics["f1"],
            "alerts_calibrated": calibrated_metrics["alerts_issued"],
            "recall_calibrated": calibrated_metrics["recall"],
            "f1_calibrated": calibrated_metrics["f1"],
            "false_positives_calibrated": calibrated_metrics["false_positive"],
        })

    monthly_df = pd.DataFrame(monthly_results)

    # Save 2024 evaluation predictions.
    validation_output = validation[
        [
            "date",
            "target_start_date",
            "target_end_date",
            "city",
            "ward_id",
            "actual_cases",
        ]
    ].copy()

    validation_output["predicted_cases"] = predicted_2024

    validation_output["alert_at_28"] = (
        predicted_2024 >= ACTUAL_HIGH_THRESHOLD
    )

    validation_output["alert_at_calibrated_threshold"] = (
        predicted_2024 >= calibrated_threshold
    )

    validation_output.to_csv(
        OUTPUT_2024,
        index=False,
    )

    # --------------------------------------------------------
    # 7. SAVE REPORT
    # --------------------------------------------------------

    report = {
        "experiment": "D - Alert threshold calibration",
        "regression_model": "Experiment A - HistGradientBoostingRegressor",
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "actual_high_incidence_threshold": ACTUAL_HIGH_THRESHOLD,
        "candidate_prediction_thresholds": [
            float(x) for x in CANDIDATE_THRESHOLDS
        ],
        "selected_prediction_alert_threshold": calibrated_threshold,
        "threshold_selection_method": (
            "Maximum F1 on 2023 calibration predictions; "
            "ties resolved by higher precision, then higher threshold"
        ),
        "training_records": len(train_df),
        "calibration_records": len(calibration_df),
        "validation_records": len(validation),
        "training_period": [
            str(train_df["date"].min().date()),
            str(train_df["date"].max().date()),
        ],
        "calibration_period": [
            str(calibration_df["date"].min().date()),
            str(calibration_df["date"].max().date()),
        ],
        "validation_period": [
            str(validation["date"].min().date()),
            str(validation["date"].max().date()),
        ],
        "calibration_results_2023": {
            "threshold_28": calibration_at_28,
            "calibrated_threshold": calibration_at_selected,
        },
        "evaluation_results_2024": {
            "threshold_28": evaluation_at_28,
            "calibrated_threshold": evaluation_at_selected,
        },
        "monthly_evaluation_2024": monthly_results,
        "test_set_used": False,
        "existing_model_modified": False,
        "existing_test_predictions_modified": False,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # 8. PRINT MONTHLY RESULTS
    # --------------------------------------------------------

    print("\n2024 MONTHLY ALERT PERFORMANCE")
    print("-" * 105)

    print(
        monthly_df.to_string(
            index=False,
        )
    )

    print("\nSAVED FILES")
    print("-" * 85)
    print("2023 calibration:", OUTPUT_2023)
    print("2024 evaluation:", OUTPUT_2024)
    print("Diagnostics:", OUTPUT_JSON)

    print("\nEXPERIMENT D COMPLETED.")
    print("No existing model or 2025 test predictions were modified.")


if __name__ == "__main__":
    main()
