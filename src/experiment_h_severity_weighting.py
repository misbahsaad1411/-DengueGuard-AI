
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "dengue_features.csv"
REPORT_DIR = ROOT / "reports"

TARGET = "target_cases_next_2_4w"

TRAIN_END = "2024-01-01"
VALIDATION_END = "2025-01-01"

HIGH_EVENT_THRESHOLD = 28
SEVERE_THRESHOLD = 45
EXTREME_THRESHOLD = 60

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

EXPERIMENTS = {
    "H0_unweighted": "unweighted",
    "H1_weight_2x_45plus": "2x for actual cases >= 45",
    "H2_weight_3x_45plus": "3x for actual cases >= 45",
    "H3_nested_severity": "2x for >= 45 and 3x for >= 60",
}


def prepare_data():
    print("=" * 95)
    print("DENGUEGUARD AI - EXPERIMENT H")
    print("SEVERITY-WEIGHTED MODEL TRAINING")
    print("=" * 95)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

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

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
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
    y_train = train_df[TARGET].astype(float).copy()

    X_validation = validation_df[feature_columns].copy()
    y_validation = validation_df[TARGET].astype(float).copy()

    print("\nDATA SUMMARY")
    print("-" * 95)
    print(f"Total records: {len(df):,}")
    print(f"Total features: {len(feature_columns)}")
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

    print("\nTRAINING TARGET DISTRIBUTION")
    print("-" * 95)
    print(f"Mean: {y_train.mean():.3f}")
    print(f"Maximum: {y_train.max():.3f}")
    print(f"45+ cases: {(y_train >= SEVERE_THRESHOLD).sum():,}")
    print(f"60+ cases: {(y_train >= EXTREME_THRESHOLD).sum():,}")

    return (
        df,
        train_df,
        validation_df,
        X_train,
        y_train,
        X_validation,
        y_validation,
        numeric_features,
    )


def build_model(numeric_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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

    preprocessor = ColumnTransformer(
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

    regressor = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    return model


def create_sample_weights(y_train, experiment):
    weights = np.ones(len(y_train), dtype=float)

    if experiment == "H0_unweighted":
        pass

    elif experiment == "H1_weight_2x_45plus":
        weights[y_train >= SEVERE_THRESHOLD] = 2.0

    elif experiment == "H2_weight_3x_45plus":
        weights[y_train >= SEVERE_THRESHOLD] = 3.0

    elif experiment == "H3_nested_severity":
        weights[y_train >= SEVERE_THRESHOLD] = 2.0
        weights[y_train >= EXTREME_THRESHOLD] = 3.0

    else:
        raise ValueError(
            f"Unknown experiment: {experiment}"
        )

    return weights


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - actual

    return {
        "records": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(errors)),
        "underprediction_records": int(
            np.sum(predicted < actual)
        ),
        "overprediction_records": int(
            np.sum(predicted > actual)
        ),
        "underprediction_rate": float(
            np.mean(predicted < actual)
        ),
        "actual_mean": float(np.mean(actual)),
        "predicted_mean": float(np.mean(predicted)),
        "actual_max": float(np.max(actual)),
        "predicted_max": float(np.max(predicted)),
    }


def calculate_severity_metrics(actual, predicted):
    severity_groups = [
        ("Below 28", actual < 28),
        ("28-34", (actual >= 28) & (actual < 35)),
        ("35-44", (actual >= 35) & (actual < 45)),
        ("45-59", (actual >= 45) & (actual < 60)),
        ("60+", actual >= 60),
    ]

    results = []

    for severity_name, mask in severity_groups:
        count = int(mask.sum())

        if count == 0:
            results.append(
                {
                    "severity": severity_name,
                    "records": 0,
                    "actual_mean": None,
                    "predicted_mean": None,
                    "bias": None,
                    "mae": None,
                    "rmse": None,
                    "underprediction_rate": None,
                    "detected_at_28": None,
                }
            )
            continue

        group_actual = actual[mask]
        group_predicted = predicted[mask]

        results.append(
            {
                "severity": severity_name,
                "records": count,
                "actual_mean": float(group_actual.mean()),
                "predicted_mean": float(group_predicted.mean()),
                "bias": float(
                    np.mean(group_predicted - group_actual)
                ),
                "mae": float(
                    mean_absolute_error(
                        group_actual,
                        group_predicted,
                    )
                ),
                "rmse": float(
                    np.sqrt(
                        mean_squared_error(
                            group_actual,
                            group_predicted,
                        )
                    )
                ),
                "underprediction_rate": float(
                    np.mean(group_predicted < group_actual)
                ),
                "detected_at_28": int(
                    np.sum(
                        group_predicted >= HIGH_EVENT_THRESHOLD
                    )
                ),
            }
        )

    return results


def calculate_alert_metrics(actual, predicted):
    actual_high = actual >= HIGH_EVENT_THRESHOLD
    predicted_high = predicted >= HIGH_EVENT_THRESHOLD

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

    actual_extreme = actual >= SEVERE_THRESHOLD

    extreme_detected = int(
        np.sum(
            actual_extreme
            & (predicted >= HIGH_EVENT_THRESHOLD)
        )
    )

    extreme_total = int(actual_extreme.sum())

    extreme_recall = (
        extreme_detected / extreme_total
        if extreme_total > 0
        else None
    )

    return {
        "alert_threshold": HIGH_EVENT_THRESHOLD,
        "actual_high_events": int(actual_high.sum()),
        "predicted_high_events": int(predicted_high.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "missed_events": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "extreme_actual_events_45plus": extreme_total,
        "extreme_events_detected_at_28": extreme_detected,
        "extreme_event_recall": (
            float(extreme_recall)
            if extreme_recall is not None
            else None
        ),
    }


def run_experiment(
    experiment_name,
    description,
    X_train,
    y_train,
    X_validation,
    y_validation,
    numeric_features,
):
    print("\n" + "=" * 95)
    print(experiment_name)
    print(description)
    print("=" * 95)

    weights = create_sample_weights(
        y_train,
        experiment_name,
    )

    model = build_model(numeric_features)

    model.fit(
        X_train,
        y_train,
        regressor__sample_weight=weights,
    )

    predictions = model.predict(X_validation)
    predictions = np.maximum(predictions, 0)

    overall_metrics = calculate_metrics(
        y_validation,
        predictions,
    )

    alert_metrics = calculate_alert_metrics(
        y_validation.to_numpy(),
        predictions,
    )

    severity_metrics = calculate_severity_metrics(
        y_validation.to_numpy(),
        predictions,
    )

    print("\nOVERALL VALIDATION PERFORMANCE")
    print("-" * 95)
    print(f"MAE: {overall_metrics['mae']:.4f}")
    print(f"RMSE: {overall_metrics['rmse']:.4f}")
    print(f"R2: {overall_metrics['r2']:.4f}")
    print(f"Bias: {overall_metrics['bias']:+.4f}")

    print("\nHIGH-INCIDENCE DETECTION")
    print("-" * 95)
    print(f"Precision: {alert_metrics['precision']:.4f}")
    print(f"Recall: {alert_metrics['recall']:.4f}")
    print(f"F1: {alert_metrics['f1']:.4f}")
    print(f"True positives: {alert_metrics['true_positive']}")
    print(f"False positives: {alert_metrics['false_positive']}")
    print(f"Missed events: {alert_metrics['missed_events']}")

    print("\nSEVERITY-BASED PERFORMANCE")
    print("-" * 95)

    for item in severity_metrics:
        print(
            f"{item['severity']:12s} "
            f"n={item['records']:4d} "
            f"MAE={item['mae'] if item['mae'] is not None else 'N/A'} "
            f"Bias={item['bias'] if item['bias'] is not None else 'N/A'} "
            f"Underprediction="
            f"{item['underprediction_rate'] if item['underprediction_rate'] is not None else 'N/A'}"
        )

    return {
        "experiment": experiment_name,
        "description": description,
        "overall": overall_metrics,
        "alerts": alert_metrics,
        "severity": severity_metrics,
        "predictions": predictions,
    }


def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        df,
        train_df,
        validation_df,
        X_train,
        y_train,
        X_validation,
        y_validation,
        numeric_features,
    ) = prepare_data()

    results = {}

    for experiment_name, description in EXPERIMENTS.items():
        results[experiment_name] = run_experiment(
            experiment_name,
            description,
            X_train,
            y_train,
            X_validation,
            y_validation,
            numeric_features,
        )

    prediction_output = validation_df[
        [
            "date",
            "target_start_date",
            "target_end_date",
            "city",
            "ward_id",
            TARGET,
        ]
    ].copy()

    prediction_output = prediction_output.rename(
        columns={
            TARGET: "actual_cases",
        }
    )

    summary_rows = []
    severity_rows = []

    diagnostics = {
        "experiment": "H",
        "title": "Severity-weighted model training",
        "training_records": int(len(train_df)),
        "validation_records": int(len(validation_df)),
        "training_period": {
            "start": str(train_df["date"].min().date()),
            "end": str(train_df["date"].max().date()),
        },
        "validation_period": {
            "start": str(validation_df["date"].min().date()),
            "end": str(validation_df["date"].max().date()),
        },
        "target": TARGET,
        "features": int(len(numeric_features) + len(CATEGORICAL_FEATURES)),
        "high_event_threshold": HIGH_EVENT_THRESHOLD,
        "severe_threshold": SEVERE_THRESHOLD,
        "extreme_threshold": EXTREME_THRESHOLD,
        "training_severe_records": int(
            (y_train >= SEVERE_THRESHOLD).sum()
        ),
        "training_extreme_records": int(
            (y_train >= EXTREME_THRESHOLD).sum()
        ),
        "results": {},
    }

    for experiment_name, result in results.items():
        predictions = result["predictions"]

        prediction_output[experiment_name] = predictions

        overall = result["overall"]
        alerts = result["alerts"]

        summary_rows.append(
            {
                "experiment": experiment_name,
                "description": result["description"],
                "mae": overall["mae"],
                "rmse": overall["rmse"],
                "r2": overall["r2"],
                "bias": overall["bias"],
                "underprediction_rate": overall[
                    "underprediction_rate"
                ],
                "high_event_precision": alerts["precision"],
                "high_event_recall": alerts["recall"],
                "high_event_f1": alerts["f1"],
                "high_event_missed": alerts["missed_events"],
                "extreme_event_recall": alerts[
                    "extreme_event_recall"
                ],
            }
        )

        for severity_item in result["severity"]:
            severity_rows.append(
                {
                    "experiment": experiment_name,
                    **severity_item,
                }
            )

        diagnostics["results"][experiment_name] = {
            "description": result["description"],
            "overall": overall,
            "alerts": alerts,
            "severity": result["severity"],
        }

    prediction_path = (
        REPORT_DIR / "experiment_h_predictions.csv"
    )

    summary_path = (
        REPORT_DIR / "experiment_h_summary.csv"
    )

    severity_path = (
        REPORT_DIR / "experiment_h_severity.csv"
    )

    diagnostics_path = (
        REPORT_DIR / "experiment_h_diagnostics.json"
    )

    prediction_output.to_csv(
        prediction_path,
        index=False,
    )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    severity_df = pd.DataFrame(severity_rows)

    severity_df.to_csv(
        severity_path,
        index=False,
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
    print("EXPERIMENT H - FINAL COMPARISON")
    print("=" * 95)

    display_columns = [
        "experiment",
        "mae",
        "rmse",
        "r2",
        "bias",
        "high_event_precision",
        "high_event_recall",
        "high_event_f1",
        "high_event_missed",
        "extreme_event_recall",
    ]

    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nFILES SAVED")
    print("-" * 95)
    print(prediction_path)
    print(summary_path)
    print(severity_path)
    print(diagnostics_path)

    print("\nExperiment H completed successfully.")
    print("The trained production model was not modified.")
    print("The 2025 test set was not used.")


if __name__ == "__main__":
    main()
