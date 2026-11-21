
from pathlib import Path
import json
import gc

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed" / "dengue_features.csv"
)

HISTORICAL_PATH = (
    ROOT / "reports" / "experiment_m_predictions.csv"
)

REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_cases_next_2_4w"

TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-01-01")

SEVERE_THRESHOLD = 45.0
CLASSIFIER_CUTOFF = 0.50

RANDOM_STATE = 42

FEATURE_EXCLUSIONS = [
    "date",
    "dengue_cases",
    TARGET,
    "target_start_date",
    "target_end_date",
]


# ============================================================
# MODEL BUILDERS
# ============================================================

def create_preprocessor():

    categorical = ["city", "ward_id"]

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            )
        ],
        remainder="passthrough",
    )


def create_forecaster(quantile=None):

    if quantile is None:

        regressor = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )

    else:

        regressor = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )

    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", regressor),
    ])


def create_warning_classifier():

    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=4,
        max_features=0.8,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", classifier),
    ])


# ============================================================
# METRIC HELPERS
# ============================================================

def regression_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "records": int(len(actual)),
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "R2": float(
            r2_score(actual, predicted)
        ),
        "bias": float(
            np.mean(predicted - actual)
        ),
    }


def event_metrics(actual, predicted, scores):

    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    scores = np.asarray(scores, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[False, True],
    ).ravel()

    return {
        "records": int(len(actual)),
        "actual_events": int(actual.sum()),
        "predicted_events": int(predicted.sum()),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "precision": float(
            precision_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "average_precision": (
            float(
                average_precision_score(
                    actual,
                    scores,
                )
            )
            if actual.sum() > 0
            else None
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 95)
    print("DENGUEGUARD AI - EXPERIMENT P")
    print("FINAL MODEL SELECTION AND HOLDOUT EVALUATION")
    print("=" * 95)

    # ========================================================
    # PART 1: HISTORICAL MODEL COMPARISON
    # ========================================================

    print("\nPART 1 - HISTORICAL MODEL COMPARISON")
    print("-" * 95)

    historical = pd.read_csv(
        HISTORICAL_PATH
    )

    required_historical = [
        "actual",
        "model_A",
        "Q50",
        "A_50_Q50_50",
    ]

    missing = [
        column
        for column in required_historical
        if column not in historical.columns
    ]

    if missing:
        raise ValueError(
            f"Missing historical columns: {missing}"
        )

    historical_results = []

    for model_name in [
        "model_A",
        "Q50",
        "A_50_Q50_50",
    ]:

        metrics = regression_metrics(
            historical["actual"],
            historical[model_name],
        )

        historical_results.append({
            "model": model_name,
            **metrics,
        })

    historical_results_df = pd.DataFrame(
        historical_results
    )

    print(
        historical_results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    historical_results_df.to_csv(
        REPORT_DIR / "experiment_p_historical_comparison.csv",
        index=False,
    )

    # ========================================================
    # PART 2: HISTORICAL WARNING MODEL COMPARISON
    # ========================================================

    print("\nPART 2 - HISTORICAL HIGH-EVENT WARNING COMPARISON")
    print("-" * 95)

    warning_predictions = pd.read_csv(
        REPORT_DIR / "experiment_o_predictions.csv"
    )

    warning_metrics = pd.read_csv(
        REPORT_DIR / "experiment_o_metrics.csv"
    )

    # Aggregate Model A and Q50 using each year's
    # own training-derived high-event threshold.

    historical_warning_results = []

    for model_name in ["model_A", "Q50"]:

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        for year, group in warning_predictions.groupby("date"):

            pass

        # Use Experiment O's fold-specific baseline metrics.
        baseline_rows = warning_metrics[
            (warning_metrics["model"] == model_name)
            & (warning_metrics["event_type"] == "high_baseline")
        ]

        total_tp = int(baseline_rows["tp"].sum())
        total_fp = int(baseline_rows["fp"].sum())
        total_fn = int(baseline_rows["fn"].sum())
        total_tn = int(baseline_rows["tn"].sum())

        precision = (
            total_tp / (total_tp + total_fp)
            if total_tp + total_fp > 0 else 0
        )

        recall = (
            total_tp / (total_tp + total_fn)
            if total_tp + total_fn > 0 else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0
        )

        historical_warning_results.append({
            "model": model_name,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "TN": total_tn,
            "precision": precision,
            "recall": recall,
            "F1": f1,
        })

    # Classifier at probability threshold 0.50.
    classifier_rows = warning_metrics[
        (warning_metrics["model"] == "high_classifier")
        & (warning_metrics["event_type"] == "high_event")
        & (warning_metrics["probability_cutoff"] == 0.50)
    ]

    tp = int(classifier_rows["tp"].sum())
    fp = int(classifier_rows["fp"].sum())
    fn = int(classifier_rows["fn"].sum())
    tn = int(classifier_rows["tn"].sum())

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    historical_warning_results.append({
        "model": "high_classifier_0.50",
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    })

    historical_warning_df = pd.DataFrame(
        historical_warning_results
    )

    print(
        historical_warning_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    historical_warning_df.to_csv(
        REPORT_DIR / "experiment_p_historical_warning_comparison.csv",
        index=False,
    )

    # ========================================================
    # PART 3: LOAD FINAL TRAINING AND TEST DATA
    # ========================================================

    print("\nPART 3 - FINAL CHRONOLOGICAL DATA SPLIT")
    print("-" * 95)

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

    train_mask = (
        (df["date"] < TEST_START)
        & (df["target_end_date"] < TEST_START)
    )

    test_mask = (
        (df["date"] >= TEST_START)
        & (df["date"] < TEST_END)
        & (df["target_end_date"] < TEST_END)
    )

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Training or 2025 test data is empty."
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in FEATURE_EXCLUSIONS
    ]

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    y_train = train_df[TARGET].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=float)

    print(f"Training records: {len(train_df):,}")
    print(f"Test records: {len(test_df):,}")
    print(
        f"Training period: "
        f"{train_df['date'].min().date()} to "
        f"{train_df['date'].max().date()}"
    )
    print(
        f"Test period: "
        f"{test_df['date'].min().date()} to "
        f"{test_df['date'].max().date()}"
    )
    print(f"Feature count: {len(feature_columns)}")

    # ========================================================
    # PART 4: TRAIN FINAL REGRESSION MODELS
    # ========================================================

    print("\nPART 4 - TRAIN FINAL REGRESSION MODELS")
    print("-" * 95)

    print("Training Model A...")

    model_a = create_forecaster()

    model_a.fit(
        X_train,
        y_train,
    )

    print("Training Q50...")

    q50_model = create_forecaster(
        quantile=0.50
    )

    q50_model.fit(
        X_train,
        y_train,
    )

    print("Generating 2025 predictions...")

    prediction_a = np.clip(
        model_a.predict(X_test),
        0,
        None,
    )

    prediction_q50 = np.clip(
        q50_model.predict(X_test),
        0,
        None,
    )

    prediction_blend = (
        0.50 * prediction_a
        + 0.50 * prediction_q50
    )

    prediction_a = np.round(prediction_a, 2)
    prediction_q50 = np.round(prediction_q50, 2)
    prediction_blend = np.round(prediction_blend, 2)

    regression_results = []

    for name, prediction in [
        ("Model_A", prediction_a),
        ("Q50", prediction_q50),
        ("50_50_Blend", prediction_blend),
    ]:

        metrics = regression_metrics(
            y_test,
            prediction,
        )

        regression_results.append({
            "model": name,
            **metrics,
        })

    regression_results_df = pd.DataFrame(
        regression_results
    )

    print("\n2025 REGRESSION RESULTS")
    print(
        regression_results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    regression_results_df.to_csv(
        REPORT_DIR / "experiment_p_2025_regression.csv",
        index=False,
    )

    # ========================================================
    # PART 5: TRAIN FINAL HIGH-EVENT WARNING CLASSIFIER
    # ========================================================

    print("\nPART 5 - TRAIN FINAL HIGH-EVENT CLASSIFIER")
    print("-" * 95)

    high_threshold = float(
        np.quantile(y_train, 0.90)
    )

    y_train_high = (
        y_train >= high_threshold
    ).astype(int)

    y_test_high = (
        y_test >= high_threshold
    )

    print(
        f"Training-derived high-event threshold: "
        f"{high_threshold:.2f}"
    )

    print(
        f"Training high events: "
        f"{y_train_high.sum():,}"
    )

    print(
        f"2025 actual high events: "
        f"{y_test_high.sum():,}"
    )

    high_classifier = create_warning_classifier()

    high_classifier.fit(
        X_train,
        y_train_high,
    )

    high_probability = high_classifier.predict_proba(
        X_test
    )[:, 1]

    high_prediction = (
        high_probability >= CLASSIFIER_CUTOFF
    )

    classifier_metrics = event_metrics(
        y_test_high,
        high_prediction,
        high_probability,
    )

    print("\n2025 HIGH-EVENT CLASSIFIER")
    print(
        json.dumps(
            classifier_metrics,
            indent=4,
        )
    )

    # ========================================================
    # PART 6: REGRESSION-BASED HIGH-EVENT COMPARISON
    # ========================================================

    print("\nPART 6 - HIGH-EVENT BASELINE COMPARISON")
    print("-" * 95)

    high_baseline_results = []

    for name, prediction in [
        ("Model_A", prediction_a),
        ("Q50", prediction_q50),
        ("50_50_Blend", prediction_blend),
    ]:

        metrics = event_metrics(
            y_test_high,
            prediction >= high_threshold,
            prediction,
        )

        high_baseline_results.append({
            "model": name,
            **metrics,
        })

    high_baseline_df = pd.DataFrame(
        high_baseline_results
    )

    print(
        high_baseline_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    high_baseline_df.to_csv(
        REPORT_DIR / "experiment_p_2025_high_event.csv",
        index=False,
    )

    # ========================================================
    # PART 7: SEVERE-EVENT DIAGNOSTICS
    # ========================================================

    print("\nPART 7 - 2025 SEVERE-EVENT DIAGNOSTICS")
    print("-" * 95)

    y_test_severe = (
        y_test >= SEVERE_THRESHOLD
    )

    severe_results = []

    for name, prediction in [
        ("Model_A", prediction_a),
        ("Q50", prediction_q50),
        ("50_50_Blend", prediction_blend),
    ]:

        metrics = event_metrics(
            y_test_severe,
            prediction >= SEVERE_THRESHOLD,
            prediction,
        )

        severe_results.append({
            "model": name,
            **metrics,
        })

    severe_df = pd.DataFrame(
        severe_results
    )

    print(
        severe_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    severe_df.to_csv(
        REPORT_DIR / "experiment_p_2025_severe_diagnostics.csv",
        index=False,
    )

    # ========================================================
    # PART 8: SAVE TEST PREDICTIONS
    # ========================================================

    print("\nPART 8 - SAVING 2025 PREDICTIONS")
    print("-" * 95)

    output = test_df[
        [
            "date",
            "target_start_date",
            "target_end_date",
            "city",
            "ward_id",
        ]
    ].copy()

    output["actual"] = y_test
    output["model_A"] = prediction_a
    output["Q50"] = prediction_q50
    output["blend_50_50"] = prediction_blend

    output["high_event_threshold"] = high_threshold
    output["high_event_probability"] = high_probability
    output["high_event_warning"] = high_prediction

    output["actual_high_event"] = y_test_high
    output["actual_severe_event"] = y_test_severe

    output.to_csv(
        REPORT_DIR / "experiment_p_2025_predictions.csv",
        index=False,
    )

    # ========================================================
    # PART 9: SAVE FINAL MODEL ARTIFACTS
    # ========================================================

    print("\nPART 9 - SAVING FINAL MODEL ARTIFACTS")
    print("-" * 95)

    forecast_artifact = {
        "model": q50_model,
        "feature_columns": feature_columns,
        "target": TARGET,
        "model_name": "Q50 Quantile HistGradientBoostingRegressor",
        "quantile": 0.50,
        "training_cutoff": "2024-12-31",
        "forecast_horizon": "Weeks 2-4",
    }

    classifier_artifact = {
        "model": high_classifier,
        "feature_columns": feature_columns,
        "target": "high_event",
        "high_event_threshold": high_threshold,
        "probability_cutoff": CLASSIFIER_CUTOFF,
        "training_cutoff": "2024-12-31",
    }

    joblib.dump(
        forecast_artifact,
        MODEL_DIR / "dengueguard_q50_forecaster.joblib",
    )

    joblib.dump(
        classifier_artifact,
        MODEL_DIR / "dengueguard_high_warning.joblib",
    )

    metadata = {
        "project": "DengueGuard AI",
        "final_forecaster": "Q50",
        "high_warning_model": "RandomForestClassifier",
        "high_warning_probability_cutoff": CLASSIFIER_CUTOFF,
        "high_event_threshold": high_threshold,
        "severe_event_threshold": SEVERE_THRESHOLD,
        "training_records": int(len(train_df)),
        "feature_count": int(len(feature_columns)),
        "training_cutoff": "2024-12-31",
        "evaluation_period": "2025",
        "forecast_horizon": "Weeks 2-4",
        "regression_test_results": regression_results,
        "high_event_test_results": classifier_metrics,
        "note": (
            "2025 was not used for fitting or model selection. "
            "This evaluation is a final holdout check."
        ),
    }

    with open(
        MODEL_DIR / "dengueguard_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # ========================================================
    # PART 10: COMPLETE
    # ========================================================

    print("\n" + "=" * 95)
    print("EXPERIMENT P COMPLETE")
    print("=" * 95)

    print("\nFINAL MODEL FILES")
    print(
        MODEL_DIR / "dengueguard_q50_forecaster.joblib"
    )
    print(
        MODEL_DIR / "dengueguard_high_warning.joblib"
    )
    print(
        MODEL_DIR / "dengueguard_metadata.json"
    )

    print("\nREPORT FILES")
    print(
        REPORT_DIR / "experiment_p_historical_comparison.csv"
    )
    print(
        REPORT_DIR / "experiment_p_historical_warning_comparison.csv"
    )
    print(
        REPORT_DIR / "experiment_p_2025_regression.csv"
    )
    print(
        REPORT_DIR / "experiment_p_2025_high_event.csv"
    )
    print(
        REPORT_DIR / "experiment_p_2025_severe_diagnostics.csv"
    )
    print(
        REPORT_DIR / "experiment_p_2025_predictions.csv"
    )

    print("\nDo not tune the models using the 2025 results.")

    del model_a, q50_model, high_classifier
    gc.collect()


if __name__ == "__main__":
    main()
