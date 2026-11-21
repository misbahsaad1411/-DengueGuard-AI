
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"

INPUT_PATH = REPORT_DIR / "experiment_l_predictions.csv"

SUMMARY_PATH = REPORT_DIR / "experiment_m_summary.json"
PREDICTIONS_PATH = REPORT_DIR / "experiment_m_predictions.csv"
YEAR_REPORT_PATH = REPORT_DIR / "experiment_m_yearly_comparison.csv"
SEVERITY_PATH = REPORT_DIR / "experiment_m_severity_analysis.csv"

MODEL_A_WEIGHTS = [1.00, 0.75, 0.50, 0.25, 0.00]

SEVERE_THRESHOLD = 45.0


# ============================================================
# METRICS
# ============================================================

def regression_metrics(actual, predicted):

    return {
        "records": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
    }


def event_metrics(actual, predicted, thresholds):

    actual_high = actual >= thresholds
    predicted_high = predicted >= thresholds

    return {
        "actual_high": int(actual_high.sum()),
        "predicted_high": int(predicted_high.sum()),
        "tp": int((actual_high & predicted_high).sum()),
        "fp": int((~actual_high & predicted_high).sum()),
        "fn": int((actual_high & ~predicted_high).sum()),
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


def severe_metrics(actual, predicted):

    mask = actual >= SEVERE_THRESHOLD

    if not mask.any():
        return {
            "records": 0,
            "mae": None,
            "bias": None,
            "underprediction_rate": None,
        }

    severe_actual = actual[mask]
    severe_predicted = predicted[mask]

    return {
        "records": int(mask.sum()),
        "mae": float(
            mean_absolute_error(
                severe_actual,
                severe_predicted,
            )
        ),
        "bias": float(
            np.mean(
                severe_predicted - severe_actual
            )
        ),
        "underprediction_rate": float(
            np.mean(
                severe_predicted < severe_actual
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 95)
    print("DENGUEGUARD AI - EXPERIMENT M")
    print("MODEL BLENDING AND OUTBREAK SENSITIVITY")
    print("=" * 95)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Experiment L predictions not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "actual",
        "model_A",
        "Q50",
        "prediction_year",
        "high_event_threshold",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=required_columns
    ).copy()

    print("\nDATA SUMMARY")
    print("-" * 95)
    print(f"Records: {len(df):,}")
    print(
        f"Validation years: "
        f"{sorted(df['prediction_year'].unique().tolist())}"
    )

    actual = df["actual"].to_numpy(dtype=float)
    model_a = df["model_A"].to_numpy(dtype=float)
    q50 = df["Q50"].to_numpy(dtype=float)

    thresholds = df[
        "high_event_threshold"
    ].to_numpy(dtype=float)

    # Each record keeps the threshold from its own
    # historical training fold.

    results = {}
    prediction_columns = []
    yearly_results = []
    severity_results = []

    # ========================================================
    # TEST BLENDING WEIGHTS
    # ========================================================

    for alpha in MODEL_A_WEIGHTS:

        beta = 1.0 - alpha

        model_name = (
            f"A_{int(alpha * 100)}_Q50_{int(beta * 100)}"
        )

        print("\n" + "=" * 95)
        print(model_name)
        print("=" * 95)

        prediction = np.maximum(
            alpha * model_a + beta * q50,
            0,
        )

        prediction = np.round(
            prediction,
            2,
        )

        regression = regression_metrics(
            actual,
            prediction,
        )

        events = event_metrics(
            actual,
            prediction,
            thresholds,
        )

        severe = severe_metrics(
            actual,
            prediction,
        )

        results[model_name] = {
            "model_A_weight": alpha,
            "Q50_weight": beta,
            "pooled_regression": regression,
            "pooled_high_event": events,
            "severe_outbreak": severe,
        }

        prediction_columns.append(
            (model_name, prediction)
        )

        print("\nPOOLED REGRESSION")
        print(json.dumps(regression, indent=2))

        print("\nPOOLED HIGH-EVENT DETECTION")
        print(json.dumps(events, indent=2))

        print("\nSEVERE OUTBREAK PERFORMANCE")
        print(json.dumps(severe, indent=2))

        # ----------------------------------------------------
        # YEAR-BY-YEAR ANALYSIS
        # ----------------------------------------------------

        for year, group in df.groupby("prediction_year"):

            y_actual = group["actual"].to_numpy(dtype=float)

            y_prediction = prediction[
                group.index.to_numpy()
            ] if group.index.equals(df.index) else None

            # Use original row positions safely.
            positions = df.index.get_indexer(group.index)
            y_prediction = prediction[positions]

            y_thresholds = group[
                "high_event_threshold"
            ].to_numpy(dtype=float)

            year_regression = regression_metrics(
                y_actual,
                y_prediction,
            )

            year_events = event_metrics(
                y_actual,
                y_prediction,
                y_thresholds,
            )

            year_severe = severe_metrics(
                y_actual,
                y_prediction,
            )

            yearly_results.append({
                "year": int(year),
                "model": model_name,
                **{
                    f"regression_{k}": v
                    for k, v in year_regression.items()
                },
                **{
                    f"event_{k}": v
                    for k, v in year_events.items()
                },
                **{
                    f"severe_{k}": v
                    for k, v in year_severe.items()
                },
            })

        # ----------------------------------------------------
        # SEVERITY ANALYSIS
        # ----------------------------------------------------

        severity_bands = [
            ("below_28", 0, 28),
            ("28_to_34", 28, 35),
            ("35_to_44", 35, 45),
            ("45_to_59", 45, 60),
            ("60_plus", 60, float("inf")),
        ]

        for label, lower, upper in severity_bands:

            mask = (
                (actual >= lower)
                & (actual < upper)
            )

            if not mask.any():
                continue

            band_metrics = regression_metrics(
                actual[mask],
                prediction[mask],
            )

            severity_results.append({
                "model": model_name,
                "severity_band": label,
                **band_metrics,
            })

    # ========================================================
    # PREDICTION OUTPUT
    # ========================================================

    output = df.copy()

    for name, prediction in prediction_columns:
        output[name] = prediction

    output.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # ========================================================
    # YEARLY OUTPUT
    # ========================================================

    yearly_df = pd.DataFrame(yearly_results)

    yearly_df.to_csv(
        YEAR_REPORT_PATH,
        index=False,
    )

    severity_df = pd.DataFrame(severity_results)

    severity_df.to_csv(
        SEVERITY_PATH,
        index=False,
    )

    # ========================================================
    # COMPACT COMPARISON
    # ========================================================

    comparison = []

    for name, result in results.items():

        reg = result["pooled_regression"]
        events = result["pooled_high_event"]
        severe = result["severe_outbreak"]

        comparison.append({
            "model": name,
            "mae": reg["mae"],
            "rmse": reg["rmse"],
            "r2": reg["r2"],
            "bias": reg["bias"],
            "precision": events["precision"],
            "recall": events["recall"],
            "f1": events["f1"],
            "severe_mae": severe["mae"],
            "severe_underprediction_rate":
                severe["underprediction_rate"],
        })

    comparison_df = pd.DataFrame(comparison)

    # Sort by MAE for inspection only.
    comparison_df = comparison_df.sort_values(
        "mae"
    )

    print("\n" + "=" * 95)
    print("FINAL BLENDING COMPARISON")
    print("=" * 95)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nYEARLY MAE COMPARISON")

    yearly_mae = yearly_df.pivot(
        index="year",
        columns="model",
        values="regression_mae",
    )

    print(
        yearly_mae.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nYEARLY HIGH-EVENT RECALL")

    yearly_recall = yearly_df.pivot(
        index="year",
        columns="model",
        values="event_recall",
    )

    print(
        yearly_recall.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {
        "experiment": "M - Model Blending",
        "input": str(INPUT_PATH),
        "validation_years": sorted(
            df["prediction_year"].unique().tolist()
        ),
        "threshold_method": (
            "Per-record threshold from its own training fold"
        ),
        "severe_threshold": SEVERE_THRESHOLD,
        "models": results,
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

    print("\nSAVED FILES")
    print(PREDICTIONS_PATH)
    print(SUMMARY_PATH)
    print(YEAR_REPORT_PATH)
    print(SEVERITY_PATH)

    print("\nEXPERIMENT M COMPLETE")


if __name__ == "__main__":
    main()
