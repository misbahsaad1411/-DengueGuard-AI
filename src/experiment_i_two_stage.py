
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "dengue_features.csv"
REPORT_DIR = ROOT / "reports"

TARGET = "target_cases_next_2_4w"

TRAIN_END = "2024-01-01"
VALIDATION_END = "2025-01-01"

HIGH_EVENT_THRESHOLD = 28
ALERT_PROBABILITY_THRESHOLD = 0.50

CATEGORICAL_FEATURES = [
    "city",
    "ward_id",
]

EXCLUDED_FEATURES = [
    "date",
    "dengue_cases",
    TARGET,
    "target_start_date",
    "target_end_date",
]


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data():

    print("=" * 95)
    print("DENGUEGUARD AI - EXPERIMENT I")
    print("TWO-STAGE OUTBREAK PREDICTION")
    print("=" * 95)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Python parser avoids the C-parser issue encountered earlier.
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "date",
            "target_start_date",
            "target_end_date",
        ],
        engine="python",
    )

    df = df.sort_values(
        ["date", "ward_id"]
    ).reset_index(drop=True)

    required_columns = (
        EXCLUDED_FEATURES
        + CATEGORICAL_FEATURES
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_FEATURES
    ]

    numeric_features = [
        column
        for column in feature_columns
        if column not in CATEGORICAL_FEATURES
    ]

    train_mask = (
        (df["date"] < TRAIN_END)
        & (df["target_end_date"] < TRAIN_END)
    )

    validation_mask = (
        (df["date"] >= TRAIN_END)
        & (df["date"] < VALIDATION_END)
        & (df["target_end_date"] < VALIDATION_END)
    )

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()

    X_train = train_df[feature_columns].copy()
    X_validation = validation_df[feature_columns].copy()

    y_train = train_df[TARGET].astype(float).to_numpy()
    y_validation = (
        validation_df[TARGET].astype(float).to_numpy()
    )

    # Binary outbreak labels are derived only from the target.
    y_train_event = (
        y_train >= HIGH_EVENT_THRESHOLD
    ).astype(int)

    y_validation_event = (
        y_validation >= HIGH_EVENT_THRESHOLD
    ).astype(int)

    print("\nDATA SUMMARY")
    print("-" * 95)

    print(f"Total records: {len(df):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")

    print(f"Training records: {len(train_df):,}")
    print(f"Validation records: {len(validation_df):,}")

    print(
        f"Training period: {train_df['date'].min().date()} "
        f"to {train_df['date'].max().date()}"
    )

    print(
        f"Validation period: {validation_df['date'].min().date()} "
        f"to {validation_df['date'].max().date()}"
    )

    print("\nOUTBREAK DISTRIBUTION")
    print("-" * 95)

    print(
        f"Training high events: "
        f"{int(y_train_event.sum()):,}"
    )

    print(
        f"Training non-high events: "
        f"{int((y_train_event == 0).sum()):,}"
    )

    print(
        f"Validation high events: "
        f"{int(y_validation_event.sum()):,}"
    )

    print(
        f"Validation non-high events: "
        f"{int((y_validation_event == 0).sum()):,}"
    )

    return {
        "df": df,
        "train_df": train_df,
        "validation_df": validation_df,
        "X_train": X_train,
        "X_validation": X_validation,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_train_event": y_train_event,
        "y_validation_event": y_validation_event,
        "numeric_features": numeric_features,
    }


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_preprocessor(numeric_features):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_regression_model(numeric_features):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features),
            ),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def build_classifier(numeric_features):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def regression_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "records": int(len(actual)),
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "r2": float(
            r2_score(actual, predicted)
        ),
        "bias": float(
            np.mean(predicted - actual)
        ),
        "actual_mean": float(
            np.mean(actual)
        ),
        "predicted_mean": float(
            np.mean(predicted)
        ),
        "underprediction_records": int(
            np.sum(predicted < actual)
        ),
        "overprediction_records": int(
            np.sum(predicted > actual)
        ),
    }


def alert_metrics(actual_event, predicted_probability):

    predicted_event = (
        predicted_probability
        >= ALERT_PROBABILITY_THRESHOLD
    ).astype(int)

    return {
        "probability_threshold": (
            ALERT_PROBABILITY_THRESHOLD
        ),
        "actual_events": int(
            np.sum(actual_event)
        ),
        "predicted_events": int(
            np.sum(predicted_event)
        ),
        "true_positive": int(
            np.sum(
                (actual_event == 1)
                & (predicted_event == 1)
            )
        ),
        "false_positive": int(
            np.sum(
                (actual_event == 0)
                & (predicted_event == 1)
            )
        ),
        "missed_events": int(
            np.sum(
                (actual_event == 1)
                & (predicted_event == 0)
            )
        ),
        "precision": float(
            precision_score(
                actual_event,
                predicted_event,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual_event,
                predicted_event,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                actual_event,
                predicted_event,
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                actual_event,
                predicted_event,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                actual_event,
                predicted_probability,
            )
        ),
        "average_precision": float(
            average_precision_score(
                actual_event,
                predicted_probability,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                actual_event,
                predicted_probability,
            )
        ),
    }


def severity_metrics(actual, predicted):

    groups = [
        ("Below 28", actual < 28),
        ("28-34", (actual >= 28) & (actual < 35)),
        ("35-44", (actual >= 35) & (actual < 45)),
        ("45-59", (actual >= 45) & (actual < 60)),
        ("60+", actual >= 60),
    ]

    output = []

    for name, mask in groups:

        count = int(mask.sum())

        if count == 0:
            output.append({
                "severity": name,
                "records": 0,
                "mae": None,
                "bias": None,
                "underprediction_rate": None,
            })
            continue

        group_actual = actual[mask]
        group_predicted = predicted[mask]

        output.append({
            "severity": name,
            "records": count,
            "mae": float(
                mean_absolute_error(
                    group_actual,
                    group_predicted,
                )
            ),
            "bias": float(
                np.mean(
                    group_predicted - group_actual
                )
            ),
            "underprediction_rate": float(
                np.mean(
                    group_predicted < group_actual
                )
            ),
        })

    return output


# ============================================================
# EXPERIMENT A REFERENCE MODEL
# ============================================================

def train_reference_model(data):

    print("\n" + "=" * 95)
    print("REFERENCE MODEL: EXPERIMENT A")
    print("=" * 95)

    model = build_regression_model(
        data["numeric_features"]
    )

    model.fit(
        data["X_train"],
        data["y_train"],
    )

    predictions = model.predict(
        data["X_validation"]
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    metrics = regression_metrics(
        data["y_validation"],
        predictions,
    )

    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"Bias: {metrics['bias']:+.4f}")

    return predictions, metrics


# ============================================================
# TWO-STAGE MODEL
# ============================================================

def train_two_stage_model(data):

    print("\n" + "=" * 95)
    print("TWO-STAGE MODEL")
    print("=" * 95)

    X_train = data["X_train"]
    X_validation = data["X_validation"]

    y_train = data["y_train"]
    y_validation = data["y_validation"]

    y_train_event = data["y_train_event"]

    # --------------------------------------------------------
    # STAGE 1: OUTBREAK CLASSIFIER
    # --------------------------------------------------------

    print("\nSTAGE 1: TRAINING OUTBREAK CLASSIFIER")
    print("-" * 95)

    classifier = build_classifier(
        data["numeric_features"]
    )

    classifier.fit(
        X_train,
        y_train_event,
    )

    probabilities = classifier.predict_proba(
        X_validation
    )[:, 1]

    probabilities = np.clip(
        probabilities,
        0,
        1,
    )

    print("Outbreak classifier trained.")

    # --------------------------------------------------------
    # STAGE 2A: LOW-INCIDENCE REGRESSOR
    # --------------------------------------------------------

    print("\nSTAGE 2A: TRAINING LOW-INCIDENCE REGRESSOR")
    print("-" * 95)

    low_mask = y_train < HIGH_EVENT_THRESHOLD

    low_regressor = build_regression_model(
        data["numeric_features"]
    )

    low_regressor.fit(
        X_train.loc[low_mask],
        y_train[low_mask],
    )

    print(
        f"Low-incidence training records: "
        f"{int(low_mask.sum()):,}"
    )

    low_predictions = low_regressor.predict(
        X_validation
    )

    # --------------------------------------------------------
    # STAGE 2B: HIGH-INCIDENCE REGRESSOR
    # --------------------------------------------------------

    print("\nSTAGE 2B: TRAINING HIGH-INCIDENCE REGRESSOR")
    print("-" * 95)

    high_mask = y_train >= HIGH_EVENT_THRESHOLD

    high_regressor = build_regression_model(
        data["numeric_features"]
    )

    high_regressor.fit(
        X_train.loc[high_mask],
        y_train[high_mask],
    )

    print(
        f"High-incidence training records: "
        f"{int(high_mask.sum()):,}"
    )

    high_predictions = high_regressor.predict(
        X_validation
    )

    # Predictions cannot be negative.
    low_predictions = np.maximum(
        low_predictions,
        0,
    )

    high_predictions = np.maximum(
        high_predictions,
        0,
    )

    # --------------------------------------------------------
    # COMBINE THE TWO REGRESSORS
    # --------------------------------------------------------

    # Probability-weighted expected prediction.
    expected_predictions = (
        (1 - probabilities) * low_predictions
        + probabilities * high_predictions
    )

    expected_predictions = np.maximum(
        expected_predictions,
        0,
    )

    # Also calculate a hard-routing prediction.
    # This selects one regressor based on the 0.5 probability
    # threshold. It is diagnostic, not the primary forecast.
    routed_predictions = np.where(
        probabilities >= ALERT_PROBABILITY_THRESHOLD,
        high_predictions,
        low_predictions,
    )

    routed_predictions = np.maximum(
        routed_predictions,
        0,
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    expected_metrics = regression_metrics(
        y_validation,
        expected_predictions,
    )

    routed_metrics = regression_metrics(
        y_validation,
        routed_predictions,
    )

    classifier_metrics = alert_metrics(
        data["y_validation_event"],
        probabilities,
    )

    expected_severity = severity_metrics(
        y_validation,
        expected_predictions,
    )

    routed_severity = severity_metrics(
        y_validation,
        routed_predictions,
    )

    print("\nOUTBREAK CLASSIFICATION RESULTS")
    print("-" * 95)

    for key, value in classifier_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nPROBABILITY-WEIGHTED REGRESSION RESULTS")
    print("-" * 95)

    for key, value in expected_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nHARD-ROUTED REGRESSION RESULTS")
    print("-" * 95)

    for key, value in routed_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nPROBABILITY-WEIGHTED SEVERITY PERFORMANCE")
    print("-" * 95)

    for item in expected_severity:
        print(
            f"{item['severity']:12s} "
            f"n={item['records']:4d} "
            f"MAE={item['mae']} "
            f"Bias={item['bias']} "
            f"Underprediction={item['underprediction_rate']}"
        )

    print("\nHARD-ROUTED SEVERITY PERFORMANCE")
    print("-" * 95)

    for item in routed_severity:
        print(
            f"{item['severity']:12s} "
            f"n={item['records']:4d} "
            f"MAE={item['mae']} "
            f"Bias={item['bias']} "
            f"Underprediction={item['underprediction_rate']}"
        )

    return {
        "probabilities": probabilities,
        "low_predictions": low_predictions,
        "high_predictions": high_predictions,
        "expected_predictions": expected_predictions,
        "routed_predictions": routed_predictions,
        "classifier_metrics": classifier_metrics,
        "expected_metrics": expected_metrics,
        "routed_metrics": routed_metrics,
        "expected_severity": expected_severity,
        "routed_severity": routed_severity,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(data, reference, two_stage):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_df = data["validation_df"]

    output = validation_df[
        [
            "date",
            "target_start_date",
            "target_end_date",
            "city",
            "ward_id",
            TARGET,
        ]
    ].copy()

    output = output.rename(
        columns={
            TARGET: "actual_cases",
        }
    )

    output["reference_A_prediction"] = (
        reference["predictions"]
    )

    output["outbreak_probability"] = (
        two_stage["probabilities"]
    )

    output["low_incidence_prediction"] = (
        two_stage["low_predictions"]
    )

    output["high_incidence_prediction"] = (
        two_stage["high_predictions"]
    )

    output["two_stage_expected_prediction"] = (
        two_stage["expected_predictions"]
    )

    output["two_stage_routed_prediction"] = (
        two_stage["routed_predictions"]
    )

    output["actual_high_event"] = (
        data["y_validation_event"]
    )

    output["predicted_high_event"] = (
        two_stage["probabilities"]
        >= ALERT_PROBABILITY_THRESHOLD
    ).astype(int)

    prediction_path = (
        REPORT_DIR / "experiment_i_predictions.csv"
    )

    output.to_csv(
        prediction_path,
        index=False,
    )

    comparison = pd.DataFrame([
        {
            "model": "Experiment A reference",
            **reference["metrics"],
        },
        {
            "model": "Two-stage expected prediction",
            **two_stage["expected_metrics"],
        },
        {
            "model": "Two-stage hard routing",
            **two_stage["routed_metrics"],
        },
    ])

    comparison_path = (
        REPORT_DIR / "experiment_i_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    diagnostics = {
        "experiment": "I",
        "title": "Two-stage outbreak prediction",
        "target": TARGET,
        "high_event_threshold": HIGH_EVENT_THRESHOLD,
        "alert_probability_threshold": (
            ALERT_PROBABILITY_THRESHOLD
        ),
        "training_records": int(
            len(data["train_df"])
        ),
        "validation_records": int(
            len(data["validation_df"])
        ),
        "training_high_events": int(
            data["y_train_event"].sum()
        ),
        "validation_high_events": int(
            data["y_validation_event"].sum()
        ),
        "reference_A": {
            "overall": reference["metrics"],
        },
        "two_stage": {
            "classifier": two_stage["classifier_metrics"],
            "expected_prediction": {
                "overall": two_stage["expected_metrics"],
                "severity": two_stage["expected_severity"],
            },
            "hard_routing": {
                "overall": two_stage["routed_metrics"],
                "severity": two_stage["routed_severity"],
            },
        },
        "notes": [
            "All models use training data before 2024.",
            "Validation uses 2024 only.",
            "The 2025 test period is not used.",
            "Current-week dengue_cases is excluded from features.",
            "The outbreak classifier target is actual target cases >= 28.",
            "The alert probability threshold is fixed at 0.50.",
            "No validation-based threshold optimization is performed.",
            "The probability-weighted prediction is the primary two-stage forecast.",
            "The hard-routed prediction is diagnostic.",
        ],
    }

    diagnostics_path = (
        REPORT_DIR / "experiment_i_diagnostics.json"
    )

    with open(
        diagnostics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            diagnostics,
            file,
            indent=4,
            allow_nan=False,
        )

    print("\n" + "=" * 95)
    print("EXPERIMENT I - FINAL MODEL COMPARISON")
    print("=" * 95)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nFILES SAVED")
    print("-" * 95)

    print(prediction_path)
    print(comparison_path)
    print(diagnostics_path)

    print("\nProduction model was not modified.")
    print("2025 test data was not used.")
    print("Experiment I completed successfully.")


# ============================================================
# MAIN
# ============================================================

def main():

    data = prepare_data()

    reference_predictions, reference_metrics = (
        train_reference_model(data)
    )

    reference = {
        "predictions": reference_predictions,
        "metrics": reference_metrics,
    }

    two_stage = train_two_stage_model(data)

    save_results(
        data,
        reference,
        two_stage,
    )


if __name__ == "__main__":
    main()
