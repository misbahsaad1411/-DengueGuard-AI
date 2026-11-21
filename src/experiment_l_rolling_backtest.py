
from pathlib import Path
import gc
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
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "dengue_features.csv"
REPORT_DIR = ROOT / "reports"

PREDICTIONS_PATH = REPORT_DIR / "experiment_l_predictions.csv"
SUMMARY_PATH = REPORT_DIR / "experiment_l_summary.json"
GROUP_REPORT_PATH = ROOT / "reports" / "experiment_l_group_analysis.csv"

TARGET = "target_cases_next_2_4w"

VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]

RANDOM_STATE = 42

# Used only for descriptive severity breakdowns.
SEVERITY_BANDS = [
    ("below_28", 0, 28),
    ("28_to_34", 28, 35),
    ("35_to_44", 35, 45),
    ("45_to_59", 45, 60),
    ("60_plus", 60, float("inf")),
]


# ============================================================
# MODEL PIPELINE
# ============================================================

def create_pipeline(loss):

    if loss == "quantile":

        regressor = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.50,
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )

    else:

        regressor = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["city", "ward_id"],
            )
        ],
        remainder="passthrough",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


# ============================================================
# METRICS
# ============================================================

def calculate_regression_metrics(actual, predicted):

    return {
        "records": int(len(actual)),
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "r2": float(
            r2_score(actual, predicted)
        ),
        "bias": float(
            np.mean(predicted - actual)
        ),
    }


def calculate_high_event_metrics(actual, predicted, threshold):

    actual_high = actual >= threshold
    predicted_high = predicted >= threshold

    return {
        "threshold": float(threshold),
        "actual_high_records": int(np.sum(actual_high)),
        "predicted_high_records": int(np.sum(predicted_high)),
        "true_positive": int(
            np.sum(actual_high & predicted_high)
        ),
        "false_positive": int(
            np.sum(~actual_high & predicted_high)
        ),
        "false_negative": int(
            np.sum(actual_high & ~predicted_high)
        ),
        "precision": float(
            precision_score(
                actual_high,
                predicted_high,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual_high,
                predicted_high,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                actual_high,
                predicted_high,
                zero_division=0,
            )
        ),
    }


# ============================================================
# SEVERITY LABELS
# ============================================================

def get_severity(value):

    for label, lower, upper in SEVERITY_BANDS:

        if lower <= value < upper:
            return label

    return "unknown"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "date",
            "target_start_date",
            "target_end_date",
        ],
        engine="python",
    )

    # Match original validation script ordering.
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
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    X = df[feature_columns].copy()
    y = df[TARGET].astype(float)

    return df, X, y, feature_columns


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 95)
    print("DENGUEGUARD AI - EXPERIMENT L")
    print("ROLLING-ORIGIN HISTORICAL BACKTESTING")
    print("=" * 95)

    df, X, y, feature_columns = load_data()

    print("\nDATA SUMMARY")
    print("-" * 95)
    print(f"Total records: {len(df):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Validation years: {VALIDATION_YEARS}")

    all_predictions = []
    fold_results = {}

    # ========================================================
    # EXPANDING-WINDOW BACKTEST
    # ========================================================

    for year in VALIDATION_YEARS:

        print("\n" + "=" * 95)
        print(f"BACKTEST YEAR: {year}")
        print("=" * 95)

        year_start = pd.Timestamp(f"{year}-01-01")
        next_year_start = pd.Timestamp(f"{year + 1}-01-01")

        # ----------------------------------------------------
        # TEMPORAL SPLIT
        # ----------------------------------------------------

        train_mask = (
            (df["date"] < year_start)
            & (df["target_end_date"] < year_start)
        )

        validation_mask = (
            (df["date"] >= year_start)
            & (df["date"] < next_year_start)
            & (df["target_end_date"] < next_year_start)
        )

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]

        X_val = X.loc[validation_mask]
        y_val = y.loc[validation_mask]

        validation_metadata = df.loc[
            validation_mask,
            [
                "date",
                "target_start_date",
                "target_end_date",
                "city",
                "ward_id",
            ],
        ].copy()

        actual = y_val.to_numpy()

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(
                f"Empty training or validation data for {year}"
            )

        print(f"Training records: {len(X_train):,}")
        print(f"Validation records: {len(X_val):,}")
        print(
            f"Training date range: "
            f"{df.loc[train_mask, 'date'].min().date()} to "
            f"{df.loc[train_mask, 'date'].max().date()}"
        )
        print(
            f"Validation date range: "
            f"{df.loc[validation_mask, 'date'].min().date()} to "
            f"{df.loc[validation_mask, 'date'].max().date()}"
        )

        # Training-only high-event threshold.
        high_threshold = float(
            y_train.quantile(0.90)
        )

        print(
            f"Training-derived high-event threshold: "
            f"{high_threshold:.2f}"
        )

        # ----------------------------------------------------
        # TRAIN MODEL A
        # ----------------------------------------------------

        print("\nTraining Model A...")

        model_a = create_pipeline("squared_error")

        model_a.fit(
            X_train,
            y_train,
        )

        prediction_a = np.round(
            np.maximum(
                model_a.predict(X_val),
                0,
            ),
            2,
        )

        metrics_a = calculate_regression_metrics(
            actual,
            prediction_a,
        )

        events_a = calculate_high_event_metrics(
            actual,
            prediction_a,
            high_threshold,
        )

        del model_a
        gc.collect()

        # ----------------------------------------------------
        # TRAIN Q50
        # ----------------------------------------------------

        print("Training Q50...")

        model_q50 = create_pipeline("quantile")

        model_q50.fit(
            X_train,
            y_train,
        )

        prediction_q50 = np.round(
            np.maximum(
                model_q50.predict(X_val),
                0,
            ),
            2,
        )

        metrics_q50 = calculate_regression_metrics(
            actual,
            prediction_q50,
        )

        events_q50 = calculate_high_event_metrics(
            actual,
            prediction_q50,
            high_threshold,
        )

        del model_q50
        gc.collect()

        # ----------------------------------------------------
        # PAIRED RECORD COMPARISON
        # ----------------------------------------------------

        error_a = np.abs(actual - prediction_a)
        error_q50 = np.abs(actual - prediction_q50)

        paired = {
            "q50_better_records": int(
                np.sum(error_q50 < error_a)
            ),
            "q50_worse_records": int(
                np.sum(error_q50 > error_a)
            ),
            "equal_error_records": int(
                np.sum(error_q50 == error_a)
            ),
            "q50_better_percentage": float(
                np.mean(error_q50 < error_a) * 100
            ),
        }

        mae_improvement = float(
            (
                metrics_a["mae"] - metrics_q50["mae"]
            )
            / metrics_a["mae"]
            * 100
        )

        # ----------------------------------------------------
        # SAVE FOLD RESULTS
        # ----------------------------------------------------

        fold_results[str(year)] = {
            "training_records": int(len(X_train)),
            "validation_records": int(len(X_val)),
            "training_high_event_threshold": high_threshold,
            "model_A": {
                "regression": metrics_a,
                "high_event": events_a,
            },
            "Q50": {
                "regression": metrics_q50,
                "high_event": events_q50,
            },
            "q50_mae_improvement_percent": mae_improvement,
            "paired_record_comparison": paired,
        }

        # ----------------------------------------------------
        # PREDICTION OUTPUT
        # ----------------------------------------------------

        validation_metadata["actual"] = actual
        validation_metadata["prediction_year"] = year
        validation_metadata["high_event_threshold"] = high_threshold
        validation_metadata["actual_severity"] = [
            get_severity(value) for value in actual
        ]

        validation_metadata["model_A"] = prediction_a
        validation_metadata["Q50"] = prediction_q50

        all_predictions.append(validation_metadata)

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print("\nMODEL A")
        print(json.dumps(metrics_a, indent=2))

        print("\nQ50")
        print(json.dumps(metrics_q50, indent=2))

        print("\nHIGH EVENT DETECTION")
        print("Model A:", json.dumps(events_a))
        print("Q50:", json.dumps(events_q50))

        print(
            f"\nQ50 MAE improvement: {mae_improvement:.2f}%"
        )

        print("\nPAIRED RECORD COMPARISON")
        print(json.dumps(paired, indent=2))

        del X_train, X_val, y_train, y_val
        del prediction_a, prediction_q50
        gc.collect()

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # ========================================================
    # OVERALL POOLED METRICS
    # ========================================================

    actual_all = predictions_df["actual"].to_numpy()

    pooled_a = predictions_df["model_A"].to_numpy()
    pooled_q50 = predictions_df["Q50"].to_numpy()

    overall_a = calculate_regression_metrics(
        actual_all,
        pooled_a,
    )

    overall_q50 = calculate_regression_metrics(
        actual_all,
        pooled_q50,
    )

    pooled_high_threshold = float(
        np.mean(predictions_df["high_event_threshold"])
    )

    pooled_events_a = calculate_high_event_metrics(
        actual_all,
        pooled_a,
        pooled_high_threshold,
    )

    pooled_events_q50 = calculate_high_event_metrics(
        actual_all,
        pooled_q50,
        pooled_high_threshold,
    )

    pooled_mae_improvement = float(
        (
            overall_a["mae"] - overall_q50["mae"]
        )
        / overall_a["mae"]
        * 100
    )

    # ========================================================
    # FOLD STABILITY
    # ========================================================

    improvements = [
        fold_results[str(year)]["q50_mae_improvement_percent"]
        for year in VALIDATION_YEARS
    ]

    stability = {
        "mean_fold_mae_improvement_percent": float(
            np.mean(improvements)
        ),
        "minimum_fold_mae_improvement_percent": float(
            np.min(improvements)
        ),
        "maximum_fold_mae_improvement_percent": float(
            np.max(improvements)
        ),
        "years_q50_improved_mae": int(
            np.sum(np.array(improvements) > 0)
        ),
        "total_years": len(VALIDATION_YEARS),
    }

    # ========================================================
    # GROUP ANALYSIS
    # ========================================================

    group_results = []

    group_columns = [
        ("city", "city"),
        ("year", "prediction_year"),
        ("severity", "actual_severity"),
    ]

    for group_type, column in group_columns:

        for group_value, group in predictions_df.groupby(
            column,
            observed=True,
        ):

            group_actual = group["actual"].to_numpy()

            for model_name in ["model_A", "Q50"]:

                group_predicted = group[model_name].to_numpy()

                metrics = calculate_regression_metrics(
                    group_actual,
                    group_predicted,
                )

                group_results.append({
                    "group_type": group_type,
                    "group": str(group_value),
                    "model": model_name,
                    **metrics,
                })

    group_report = pd.DataFrame(group_results)

    group_report.to_csv(
        GROUP_REPORT_PATH,
        index=False,
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {
        "experiment": "L - Rolling-Origin Backtesting",
        "validation_years": VALIDATION_YEARS,
        "random_state": RANDOM_STATE,
        "fold_results": fold_results,
        "pooled_results": {
            "model_A": {
                "regression": overall_a,
                "high_event": pooled_events_a,
            },
            "Q50": {
                "regression": overall_q50,
                "high_event": pooled_events_q50,
            },
            "q50_mae_improvement_percent": pooled_mae_improvement,
            "pooled_high_event_threshold": pooled_high_threshold,
        },
        "fold_stability": stability,
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 95)
    print("EXPERIMENT L FINAL SUMMARY")
    print("=" * 95)

    print("\nPOOLED RESULTS ACROSS ALL VALIDATION YEARS")

    print(
        f"{'Model':<15}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
        f"{'R2':>12}"
        f"{'Bias':>12}"
    )

    for name, metrics in [
        ("Model A", overall_a),
        ("Q50", overall_q50),
    ]:

        print(
            f"{name:<15}"
            f"{metrics['mae']:>12.4f}"
            f"{metrics['rmse']:>12.4f}"
            f"{metrics['r2']:>12.4f}"
            f"{metrics['bias']:>12.4f}"
        )

    print(
        f"\nPooled Q50 MAE improvement: "
        f"{pooled_mae_improvement:.2f}%"
    )

    print("\nFOLD STABILITY")
    print(json.dumps(stability, indent=2))

    print("\nHIGH EVENT COMPARISON")
    print("Model A:", json.dumps(pooled_events_a))
    print("Q50:", json.dumps(pooled_events_q50))

    print("\nSAVED FILES")
    print(PREDICTIONS_PATH)
    print(SUMMARY_PATH)
    print(GROUP_REPORT_PATH)

    print("\nEXPERIMENT L COMPLETE")


if __name__ == "__main__":
    main()
