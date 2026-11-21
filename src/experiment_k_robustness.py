
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
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "dengue_features.csv"
REPORT_DIR = ROOT / "reports"

PREDICTIONS_PATH = REPORT_DIR / "experiment_k_predictions.csv"
SUMMARY_PATH = REPORT_DIR / "experiment_k_summary.json"
GROUP_REPORT_PATH = REPORT_DIR / "experiment_k_group_analysis.csv"

TARGET = "target_cases_next_2_4w"

TRAIN_START = "2024-01-01"
VALIDATION_END = "2025-01-01"

HIGH_THRESHOLD = 28.0

# Original seed first, followed by alternative seeds.
RANDOM_SEEDS = [42, 21, 84, 123, 2026]


# ============================================================
# MODEL
# ============================================================

def create_pipeline(loss, seed):

    if loss == "quantile":
        regressor = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.50,
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        )

    else:
        regressor = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
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

def calculate_metrics(actual, predicted):

    return {
        "records": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
    }


def calculate_event_metrics(actual, predicted):

    actual_high = actual >= HIGH_THRESHOLD
    predicted_high = predicted >= HIGH_THRESHOLD

    return {
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
# GROUP ANALYSIS
# ============================================================

def group_analysis(df, prediction_columns):

    results = []

    group_definitions = [
        ("city", "city"),
        ("prediction_month", "prediction_month"),
        ("severity", "severity"),
    ]

    for group_type, column in group_definitions:

        for group_value, group in df.groupby(
            column,
            observed=True
        ):

            actual = group["actual"].to_numpy()

            for model_name in prediction_columns:

                predicted = group[model_name].to_numpy()

                metrics = calculate_metrics(
                    actual,
                    predicted
                )

                results.append({
                    "group_type": group_type,
                    "group": str(group_value),
                    "model": model_name,
                    **metrics,
                })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print("DENGUEGUARD AI - EXPERIMENT K")
    print("Q50 ROBUSTNESS AND STABILITY ANALYSIS")
    print("=" * 95)

    # --------------------------------------------------------
    # LOAD AND SORT DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "date",
            "target_start_date",
            "target_end_date",
        ],
        engine="python",
    )

    # Match original validation evaluation ordering.
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

    validation_info = df.loc[
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

    print("\nDATA SUMMARY")
    print("-" * 95)
    print(f"Training records: {len(X_train):,}")
    print(f"Validation records: {len(X_val):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Random seeds: {RANDOM_SEEDS}")

    # --------------------------------------------------------
    # SEVERITY GROUPS
    # --------------------------------------------------------

    severity = np.select(
        [
            actual < 28,
            (actual >= 28) & (actual < 35),
            (actual >= 35) & (actual < 45),
            (actual >= 45) & (actual < 60),
            actual >= 60,
        ],
        [
            "below_28",
            "28_to_34",
            "35_to_44",
            "45_to_59",
            "60_plus",
        ],
        default="unknown",
    )

    validation_info["actual"] = actual
    validation_info["prediction_month"] = (
        validation_info["date"].dt.month
    )
    validation_info["severity"] = severity

    # --------------------------------------------------------
    # TRAIN MODELS ACROSS RANDOM SEEDS
    # --------------------------------------------------------

    all_predictions = validation_info.copy()

    seed_results = {}

    for seed in RANDOM_SEEDS:

        print("\n" + "-" * 95)
        print(f"TRAINING MODELS WITH RANDOM SEED {seed}")
        print("-" * 95)

        # Model A
        model_a = create_pipeline(
            loss="squared_error",
            seed=seed,
        )

        model_a.fit(X_train, y_train)

        pred_a = np.round(
            np.maximum(model_a.predict(X_val), 0),
            2,
        )

        # Q50
        model_q50 = create_pipeline(
            loss="quantile",
            seed=seed,
        )

        model_q50.fit(X_train, y_train)

        pred_q50 = np.round(
            np.maximum(model_q50.predict(X_val), 0),
            2,
        )

        name_a = f"A_seed_{seed}"
        name_q50 = f"Q50_seed_{seed}"

        all_predictions[name_a] = pred_a
        all_predictions[name_q50] = pred_q50

        metrics_a = calculate_metrics(
            actual,
            pred_a,
        )

        metrics_q50 = calculate_metrics(
            actual,
            pred_q50,
        )

        events_a = calculate_event_metrics(
            actual,
            pred_a,
        )

        events_q50 = calculate_event_metrics(
            actual,
            pred_q50,
        )

        seed_results[str(seed)] = {
            "model_A": {
                "metrics": metrics_a,
                "high_event": events_a,
            },
            "model_Q50": {
                "metrics": metrics_q50,
                "high_event": events_q50,
            },
            "mae_improvement_percent": float(
                (
                    metrics_a["mae"] - metrics_q50["mae"]
                )
                / metrics_a["mae"]
                * 100
            ),
            "q50_better_high_event_f1": bool(
                events_q50["f1"] > events_a["f1"]
            ),
        }

        print("\nMODEL A")
        print(json.dumps(metrics_a, indent=2))

        print("\nMODEL Q50")
        print(json.dumps(metrics_q50, indent=2))

        print("\nHIGH EVENT COMPARISON")
        print("A:", json.dumps(events_a))
        print("Q50:", json.dumps(events_q50))

        print(
            "\nQ50 MAE improvement:",
            f"{seed_results[str(seed)]['mae_improvement_percent']:.2f}%"
        )

    # --------------------------------------------------------
    # MAIN PAIRED RECORD COMPARISON - SEED 42
    # --------------------------------------------------------

    a42 = all_predictions["A_seed_42"].to_numpy()
    q5042 = all_predictions["Q50_seed_42"].to_numpy()

    error_a = np.abs(actual - a42)
    error_q50 = np.abs(actual - q5042)

    paired_results = {
        "q50_lower_error_records": int(
            np.sum(error_q50 < error_a)
        ),
        "q50_higher_error_records": int(
            np.sum(error_q50 > error_a)
        ),
        "equal_error_records": int(
            np.sum(error_q50 == error_a)
        ),
        "q50_better_record_percentage": float(
            np.mean(error_q50 < error_a) * 100
        ),
        "mean_absolute_error_difference": float(
            np.mean(error_a - error_q50)
        ),
    }

    # --------------------------------------------------------
    # CITY, MONTH AND SEVERITY ANALYSIS - SEED 42
    # --------------------------------------------------------

    group_input = validation_info.copy()

    group_input["Model_A"] = a42
    group_input["Q50"] = q5042

    group_report = group_analysis(
        group_input,
        ["Model_A", "Q50"],
    )

    group_report.to_csv(
        GROUP_REPORT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # SEVERITY SUMMARY
    # --------------------------------------------------------

    severity_report = group_report[
        group_report["group_type"] == "severity"
    ]

    # --------------------------------------------------------
    # SEED STABILITY
    # --------------------------------------------------------

    improvements = [
        result["mae_improvement_percent"]
        for result in seed_results.values()
    ]

    stability = {
        "mean_mae_improvement_percent": float(
            np.mean(improvements)
        ),
        "minimum_mae_improvement_percent": float(
            np.min(improvements)
        ),
        "maximum_mae_improvement_percent": float(
            np.max(improvements)
        ),
        "seeds_where_q50_improved_mae": int(
            np.sum(np.array(improvements) > 0)
        ),
        "total_seeds": len(RANDOM_SEEDS),
    }

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    all_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary = {
        "experiment": "K - Q50 Robustness",
        "training_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "feature_count": int(len(feature_columns)),
        "high_event_threshold": HIGH_THRESHOLD,
        "random_seeds": RANDOM_SEEDS,
        "seed_results": seed_results,
        "paired_record_comparison_seed_42": paired_results,
        "seed_stability": stability,
        "severity_analysis_seed_42": (
            severity_report.to_dict(orient="records")
        ),
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

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 95)
    print("EXPERIMENT K FINAL RESULTS")
    print("=" * 95)

    print("\nSEED STABILITY")
    print(json.dumps(stability, indent=2))

    print("\nPAIRED RECORD COMPARISON - SEED 42")
    print(json.dumps(paired_results, indent=2))

    print("\nSEVERITY ANALYSIS - SEED 42")
    print(
        severity_report.to_string(index=False)
    )

    print("\nCITY AND MONTH ANALYSIS - SEED 42")
    print(
        group_report[
            group_report["group_type"].isin(
                ["city", "prediction_month"]
            )
        ].to_string(index=False)
    )

    print("\nSAVED FILES")
    print(PREDICTIONS_PATH)
    print(SUMMARY_PATH)
    print(GROUP_REPORT_PATH)

    print("\nEXPERIMENT K COMPLETE")


if __name__ == "__main__":
    main()
