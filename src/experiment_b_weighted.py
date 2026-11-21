
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# DENGUEGUARD AI - EXPERIMENT B
# HIGH-INCIDENCE WEIGHTED TRAINING
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "dengue_features.csv"
REPORT_DIR = ROOT / "reports"

PREDICTIONS_PATH = REPORT_DIR / "experiment_b_predictions.csv"
REPORT_PATH = REPORT_DIR / "experiment_b_diagnostics.json"

TARGET = "target_cases_next_2_4w"

VALIDATION_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")

CATEGORICAL_FEATURES = ["city", "ward_id"]

EXCLUDED_COLUMNS = [
    "date",
    "dengue_cases",
    TARGET,
    "target_start_date",
    "target_end_date",
]

# Predeclared weighting strengths.
# 1.0 = unweighted reference model.
WEIGHT_STRENGTHS = [1.5, 2.0, 3.0]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    """Calculate regression performance metrics."""

    actual = np.asarray(actual, dtype=float)

    predicted = np.maximum(
        np.asarray(predicted, dtype=float),
        0
    )

    if len(actual) == 0:
        return {}

    return {
        "records": int(len(actual)),

        "actual_mean": round(
            float(np.mean(actual)), 4
        ),

        "predicted_mean": round(
            float(np.mean(predicted)), 4
        ),

        "mae": round(
            float(mean_absolute_error(actual, predicted)), 4
        ),

        "rmse": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 4
        ),

        "r2": round(
            float(r2_score(actual, predicted)), 4
        ),

        "bias": round(
            float(np.mean(predicted - actual)), 4
        ),
    }


# ============================================================
# HIGH-INCIDENCE METRICS
# ============================================================

def calculate_high_incidence_metrics(
    actual,
    predicted,
    threshold
):
    """Evaluate detection performance for high-incidence observations."""

    actual = np.asarray(actual, dtype=float)

    predicted = np.maximum(
        np.asarray(predicted, dtype=float),
        0
    )

    actual_high = actual >= threshold
    predicted_high = predicted >= threshold

    true_positive = int(
        np.sum(actual_high & predicted_high)
    )

    false_positive = int(
        np.sum(~actual_high & predicted_high)
    )

    false_negative = int(
        np.sum(actual_high & ~predicted_high)
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    high_actual = actual[actual_high]
    high_predicted = predicted[actual_high]

    high_mae = (
        float(mean_absolute_error(high_actual, high_predicted))
        if len(high_actual) > 0
        else None
    )

    return {
        "threshold": round(float(threshold), 4),

        "actual_high_records": int(
            np.sum(actual_high)
        ),

        "predicted_high_records": int(
            np.sum(predicted_high)
        ),

        "true_positive": true_positive,

        "false_positive": false_positive,

        "missed_high_events": false_negative,

        "precision": round(float(precision), 4),

        "recall": round(float(recall), 4),

        "f1": round(float(f1), 4),

        "high_incidence_mae": (
            round(high_mae, 4)
            if high_mae is not None
            else None
        ),

        "high_event_missed_rate": round(
            float(1 - recall), 4
        ),
    }


# ============================================================
# MODEL PIPELINE
# SAME CONFIGURATION AS EXPERIMENT A
# ============================================================

def build_model(numeric_features):
    """Create the same model pipeline used in Experiment A."""

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

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("DENGUEGUARD AI - EXPERIMENT B")
    print("HIGH-INCIDENCE WEIGHTED TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "date",
            "target_start_date",
            "target_end_date"
        ]
    )

    # IMPORTANT:
    # Use exactly the same row ordering as Experiment A.
    df = df.sort_values(
        ["date", "ward_id"]
    ).reset_index(drop=True)

    required_columns = (
        CATEGORICAL_FEATURES
        + EXCLUDED_COLUMNS
        + ["dengue_cases_rolling_4w"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 2. FEATURE SELECTION
    # SAME AS EXPERIMENT A
    # --------------------------------------------------------

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
        raise ValueError(
            "Missing values found in model features."
        )

    # --------------------------------------------------------
    # 3. CHRONOLOGICAL DATA SPLIT
    # SAME AS EXPERIMENT A
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
        raise ValueError(
            "Training or validation dataset is empty."
        )

    X_train = train_df[feature_columns]

    y_train = train_df[TARGET]

    X_validation = validation_df[feature_columns]

    y_validation = validation_df[TARGET]

    print("\nDATA SPLIT")
    print("-" * 70)

    print("Training records:", len(train_df))

    print("Validation records:", len(validation_df))

    print("Feature count:", len(feature_columns))

    print(
        "Training period:",
        train_df["date"].min().date(),
        "to",
        train_df["date"].max().date()
    )

    print(
        "Validation period:",
        validation_df["date"].min().date(),
        "to",
        validation_df["date"].max().date()
    )

    # --------------------------------------------------------
    # 4. HIGH-INCIDENCE THRESHOLD
    # CALCULATED ONLY FROM TRAINING DATA
    # --------------------------------------------------------

    high_threshold = float(
        y_train.quantile(0.90)
    )

    high_training = y_train >= high_threshold

    print("\nHIGH-INCIDENCE TRAINING THRESHOLD")
    print("-" * 70)

    print(
        "Training 90th percentile:",
        round(high_threshold, 4)
    )

    print(
        "High-incidence training records:",
        int(high_training.sum())
    )

    # --------------------------------------------------------
    # 5. TRAIN UNWEIGHTED REFERENCE MODEL
    # --------------------------------------------------------

    print("\nTRAINING UNWEIGHTED REFERENCE MODEL...")

    reference_model = build_model(
        numeric_features
    )

    reference_model.fit(
        X_train,
        y_train
    )

    reference_predictions = np.maximum(
        reference_model.predict(X_validation),
        0
    )

    reference_overall = calculate_metrics(
        y_validation,
        reference_predictions
    )

    reference_high = calculate_high_incidence_metrics(
        y_validation,
        reference_predictions,
        high_threshold
    )

    results = {}

    results["1.0"] = {
        "experiment": "Unweighted reference",
        "overall": reference_overall,
        "high_incidence": reference_high
    }

    print("\nUNWEIGHTED REFERENCE PERFORMANCE")
    print("-" * 70)

    print(
        json.dumps(
            results["1.0"],
            indent=4
        )
    )

    # --------------------------------------------------------
    # 6. SAVE VALIDATION PREDICTIONS
    # --------------------------------------------------------

    predictions = pd.DataFrame({
        "date": validation_df["date"].values,

        "target_start_date": validation_df[
            "target_start_date"
        ].values,

        "target_end_date": validation_df[
            "target_end_date"
        ].values,

        "city": validation_df["city"].values,

        "ward_id": validation_df["ward_id"].values,

        "actual_cases": y_validation.values,

        "baseline_prediction": np.maximum(
            validation_df[
                "dengue_cases_rolling_4w"
            ].values * 3,
            0
        ),

        "prediction_weight_1_0": reference_predictions
    })

    # --------------------------------------------------------
    # 7. TRAIN WEIGHTED MODELS
    # --------------------------------------------------------

    for strength in WEIGHT_STRENGTHS:

        print("\n" + "=" * 70)

        print(
            f"TRAINING WEIGHTED MODEL: {strength}x"
        )

        print("=" * 70)

        # Start with equal weights for every observation.
        sample_weights = np.ones(
            len(y_train),
            dtype=float
        )

        # Increase importance of high-incidence observations.
        sample_weights[high_training.to_numpy()] = strength

        print(
            "High-incidence training records:",
            int(high_training.sum())
        )

        print(
            "Weight assigned to high-incidence records:",
            strength
        )

        model = build_model(
            numeric_features
        )

        model.fit(
            X_train,
            y_train,
            regressor__sample_weight=sample_weights
        )

        validation_predictions = np.maximum(
            model.predict(X_validation),
            0
        )

        overall = calculate_metrics(
            y_validation,
            validation_predictions
        )

        high_metrics = calculate_high_incidence_metrics(
            y_validation,
            validation_predictions,
            high_threshold
        )

        key = str(strength)

        results[key] = {
            "experiment": (
                f"High-incidence weighting {strength}x"
            ),

            "training_weight": strength,

            "overall": overall,

            "high_incidence": high_metrics
        }

        predictions[
            f"prediction_weight_{key.replace('.', '_')}"
        ] = validation_predictions

        print("\nOVERALL PERFORMANCE")

        print(
            json.dumps(
                overall,
                indent=4
            )
        )

        print("\nHIGH-INCIDENCE PERFORMANCE")

        print(
            json.dumps(
                high_metrics,
                indent=4
            )
        )

    # --------------------------------------------------------
    # 8. BASELINE METRICS
    # --------------------------------------------------------

    baseline_predictions = predictions[
        "baseline_prediction"
    ].values

    baseline_metrics = calculate_metrics(
        y_validation,
        baseline_predictions
    )

    # --------------------------------------------------------
    # 9. SAVE PREDICTIONS AND REPORT
    # --------------------------------------------------------

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    report = {
        "experiment": (
            "B - High-incidence weighted training"
        ),

        "target": TARGET,

        "target_definition": (
            "Sum of dengue cases at weeks t+2, t+3 and t+4"
        ),

        "training_records": int(len(train_df)),

        "validation_records": int(len(validation_df)),

        "feature_count": int(len(feature_columns)),

        "feature_columns": feature_columns,

        "high_incidence_threshold_from_training": (
            high_threshold
        ),

        "high_incidence_training_records": int(
            high_training.sum()
        ),

        "weighting_strengths_tested": WEIGHT_STRENGTHS,

        "baseline_performance": baseline_metrics,

        "results": results,

        "test_set_used": False
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # 10. FINAL COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 90)

    print("EXPERIMENT B - FINAL VALIDATION COMPARISON")

    print("=" * 90)

    print(
        f"{'Weight':<12}"
        f"{'MAE':<12}"
        f"{'RMSE':<12}"
        f"{'R2':<12}"
        f"{'Bias':<12}"
        f"{'Precision':<13}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Miss Rate':<12}"
    )

    print("-" * 105)

    for key, result in results.items():

        overall = result["overall"]

        high = result["high_incidence"]

        print(
            f"{key:<12}"
            f"{overall['mae']:<12.4f}"
            f"{overall['rmse']:<12.4f}"
            f"{overall['r2']:<12.4f}"
            f"{overall['bias']:<12.4f}"
            f"{high['precision']:<13.4f}"
            f"{high['recall']:<12.4f}"
            f"{high['f1']:<12.4f}"
            f"{high['high_event_missed_rate']:<12.4f}"
        )

    # --------------------------------------------------------
    # 11. BASELINE COMPARISON
    # --------------------------------------------------------

    print("\nBASELINE PERFORMANCE")
    print("-" * 70)

    print(
        json.dumps(
            baseline_metrics,
            indent=4
        )
    )

    # --------------------------------------------------------
    # 12. SAVED FILES
    # --------------------------------------------------------

    print("\nSAVED FILES")
    print("-" * 70)

    print("Predictions:", PREDICTIONS_PATH)

    print("Diagnostics:", REPORT_PATH)

    print("\nEXPERIMENT B COMPLETED.")

    print(
        "The existing model and 2025 test results were not modified."
    )


if __name__ == "__main__":
    main()
