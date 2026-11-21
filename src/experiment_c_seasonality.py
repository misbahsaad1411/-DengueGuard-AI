
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# DENGUEGUARD AI - EXPERIMENT C
# FUTURE-SEASON FEATURES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "dengue_features.csv"
REPORT_DIR = ROOT / "reports"

PREDICTIONS_PATH = REPORT_DIR / "experiment_c_predictions.csv"
REPORT_PATH = REPORT_DIR / "experiment_c_diagnostics.json"

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

HIGH_INCIDENCE_WEIGHT = 2.0


# ============================================================
# FUTURE-SEASON FEATURES
# ============================================================

def add_future_season_features(df):
    """
    Add calendar information for weeks t+2, t+3 and t+4.

    These features use only the prediction date.
    No future observed weather or dengue data is used.
    """

    df = df.copy()

    for offset in [2, 3, 4]:

        future_date = (
            df["date"] + pd.to_timedelta(offset * 7, unit="D")
        )

        future_month = future_date.dt.month

        angle = 2 * np.pi * (future_month - 1) / 12

        df[f"future_month_sin_t{offset}"] = np.sin(angle)

        df[f"future_month_cos_t{offset}"] = np.cos(angle)

    return df


# ============================================================
# MODEL
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

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float),
        0,
    )

    return {
        "records": int(len(actual)),
        "actual_mean": round(float(np.mean(actual)), 4),
        "predicted_mean": round(float(np.mean(predicted)), 4),
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(
            float(np.sqrt(mean_squared_error(actual, predicted))),
            4,
        ),
        "r2": round(float(r2_score(actual, predicted)), 4),
        "bias": round(float(np.mean(predicted - actual)), 4),
    }


def calculate_high_incidence_metrics(actual, predicted, threshold):

    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float),
        0,
    )

    actual_high = actual >= threshold
    predicted_high = predicted >= threshold

    tp = int(np.sum(actual_high & predicted_high))
    fp = int(np.sum(~actual_high & predicted_high))
    fn = int(np.sum(actual_high & ~predicted_high))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    high_actual = actual[actual_high]
    high_predicted = predicted[actual_high]

    high_mae = (
        float(mean_absolute_error(high_actual, high_predicted))
        if len(high_actual)
        else None
    )

    return {
        "threshold": round(float(threshold), 4),
        "actual_high_records": int(np.sum(actual_high)),
        "predicted_high_records": int(np.sum(predicted_high)),
        "true_positive": tp,
        "false_positive": fp,
        "missed_high_events": fn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "high_incidence_mae": (
            round(high_mae, 4) if high_mae is not None else None
        ),
        "high_event_missed_rate": round(float(1 - recall), 4),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("DENGUEGUARD AI - EXPERIMENT C")
    print("FUTURE-SEASON FEATURES")
    print("=" * 75)

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
            "target_end_date",
        ],
    )

    # Same ordering as Experiment A and B.
    df = df.sort_values(
        ["date", "ward_id"]
    ).reset_index(drop=True)

    required_columns = (
        CATEGORICAL_FEATURES
        + EXCLUDED_COLUMNS
        + ["dengue_cases_rolling_4w"]
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 2. ADD FUTURE-SEASON FEATURES
    # --------------------------------------------------------

    df = add_future_season_features(df)

    new_features = [
        "future_month_sin_t2",
        "future_month_cos_t2",
        "future_month_sin_t3",
        "future_month_cos_t3",
        "future_month_sin_t4",
        "future_month_cos_t4",
    ]

    # --------------------------------------------------------
    # 3. FEATURE SELECTION
    # --------------------------------------------------------

    feature_columns = [
        col for col in df.columns
        if col not in EXCLUDED_COLUMNS
    ]

    numeric_features = [
        col for col in feature_columns
        if col not in CATEGORICAL_FEATURES
    ]

    if df[feature_columns].isnull().any().any():
        raise ValueError(
            "Missing values found in model features."
        )

    # --------------------------------------------------------
    # 4. CHRONOLOGICAL SPLIT
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
    print("-" * 75)

    print("Training records:", len(train_df))
    print("Validation records:", len(validation_df))
    print("Original features: 46")
    print("New future-season features:", len(new_features))
    print("Total features:", len(feature_columns))

    print(
        "Training period:",
        train_df["date"].min().date(),
        "to",
        train_df["date"].max().date(),
    )

    print(
        "Validation period:",
        validation_df["date"].min().date(),
        "to",
        validation_df["date"].max().date(),
    )

    print("\nNEW FEATURES")
    for feature in new_features:
        print("-", feature)

    # --------------------------------------------------------
    # 5. HIGH-INCIDENCE THRESHOLD
    # --------------------------------------------------------

    high_threshold = float(y_train.quantile(0.90))

    high_training = y_train >= high_threshold

    print("\nHIGH-INCIDENCE THRESHOLD")
    print("-" * 75)
    print("Training 90th percentile:", high_threshold)
    print("High-incidence training records:", int(high_training.sum()))

    # --------------------------------------------------------
    # 6. TRAIN UNWEIGHTED FUTURE-SEASON MODEL
    # --------------------------------------------------------

    print("\nTRAINING C1: UNWEIGHTED FUTURE-SEASON MODEL...")

    model_unweighted = build_model(numeric_features)

    model_unweighted.fit(
        X_train,
        y_train,
    )

    predictions_unweighted = np.maximum(
        model_unweighted.predict(X_validation),
        0,
    )

    results = {
        "C1_unweighted": {
            "experiment": "Future-season features, unweighted",
            "overall": calculate_metrics(
                y_validation,
                predictions_unweighted,
            ),
            "high_incidence": calculate_high_incidence_metrics(
                y_validation,
                predictions_unweighted,
                high_threshold,
            ),
        }
    }

    # --------------------------------------------------------
    # 7. TRAIN 2X WEIGHTED FUTURE-SEASON MODEL
    # --------------------------------------------------------

    print("\nTRAINING C2: 2X WEIGHTED FUTURE-SEASON MODEL...")

    sample_weights = np.ones(
        len(y_train),
        dtype=float,
    )

    sample_weights[high_training.to_numpy()] = (
        HIGH_INCIDENCE_WEIGHT
    )

    model_weighted = build_model(numeric_features)

    model_weighted.fit(
        X_train,
        y_train,
        regressor__sample_weight=sample_weights,
    )

    predictions_weighted = np.maximum(
        model_weighted.predict(X_validation),
        0,
    )

    results["C2_weighted_2x"] = {
        "experiment": "Future-season features, 2x high-incidence weighting",
        "training_weight": HIGH_INCIDENCE_WEIGHT,
        "overall": calculate_metrics(
            y_validation,
            predictions_weighted,
        ),
        "high_incidence": calculate_high_incidence_metrics(
            y_validation,
            predictions_weighted,
            high_threshold,
        ),
    }

    # --------------------------------------------------------
    # 8. BASELINE
    # --------------------------------------------------------

    baseline_predictions = np.maximum(
        validation_df["dengue_cases_rolling_4w"].values * 3,
        0,
    )

    baseline_metrics = calculate_metrics(
        y_validation,
        baseline_predictions,
    )

    # --------------------------------------------------------
    # 9. SAVE PREDICTIONS
    # --------------------------------------------------------

    predictions = pd.DataFrame({
        "date": validation_df["date"].values,
        "target_start_date": validation_df["target_start_date"].values,
        "target_end_date": validation_df["target_end_date"].values,
        "city": validation_df["city"].values,
        "ward_id": validation_df["ward_id"].values,
        "actual_cases": y_validation.values,
        "baseline_prediction": baseline_predictions,
        "prediction_C1_unweighted": predictions_unweighted,
        "prediction_C2_weighted_2x": predictions_weighted,
    })

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 10. SAVE DIAGNOSTICS
    # --------------------------------------------------------

    report = {
        "experiment": "C - Future-season features",
        "target": TARGET,
        "target_definition": (
            "Sum of dengue cases at weeks t+2, t+3 and t+4"
        ),
        "feature_count": len(feature_columns),
        "original_feature_count": 46,
        "new_features": new_features,
        "feature_columns": feature_columns,
        "training_records": len(train_df),
        "validation_records": len(validation_df),
        "training_period": [
            str(train_df["date"].min().date()),
            str(train_df["date"].max().date()),
        ],
        "validation_period": [
            str(validation_df["date"].min().date()),
            str(validation_df["date"].max().date()),
        ],
        "high_incidence_threshold_from_training": high_threshold,
        "high_incidence_training_records": int(high_training.sum()),
        "baseline_performance": baseline_metrics,
        "results": results,
        "test_set_used": False,
        "existing_model_modified": False,
        "test_predictions_modified": False,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # 11. FINAL COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("EXPERIMENT C - VALIDATION COMPARISON")
    print("=" * 100)

    print(
        f"{'Model':<25}"
        f"{'MAE':<12}"
        f"{'RMSE':<12}"
        f"{'R2':<12}"
        f"{'Bias':<12}"
        f"{'Precision':<13}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Miss Rate':<12}"
    )

    print("-" * 110)

    for key, result in results.items():

        overall = result["overall"]
        high = result["high_incidence"]

        print(
            f"{key:<25}"
            f"{overall['mae']:<12.4f}"
            f"{overall['rmse']:<12.4f}"
            f"{overall['r2']:<12.4f}"
            f"{overall['bias']:<12.4f}"
            f"{high['precision']:<13.4f}"
            f"{high['recall']:<12.4f}"
            f"{high['f1']:<12.4f}"
            f"{high['high_event_missed_rate']:<12.4f}"
        )

    print("\nBASELINE")
    print(json.dumps(baseline_metrics, indent=4))

    print("\nSAVED FILES")
    print("Predictions:", PREDICTIONS_PATH)
    print("Diagnostics:", REPORT_PATH)

    print("\nEXPERIMENT C COMPLETED.")
    print("The existing model and 2025 test results were not modified.")


if __name__ == "__main__":
    main()
