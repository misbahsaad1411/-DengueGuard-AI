
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "reports" / "experiment_d_2024_evaluation.csv"
OUTPUT = ROOT / "reports" / "experiment_d_threshold_comparison_2024.csv"

ACTUAL_THRESHOLD = 28.0

# All thresholds were selected or defined without using 2024 outcomes.
THRESHOLDS = [24.0, 24.5, 27.0, 28.0, 30.0]


def evaluate(actual, predicted, threshold):

    actual_high = actual >= ACTUAL_THRESHOLD
    alerts = predicted >= threshold

    return {
        "prediction_threshold": threshold,
        "alerts": int(alerts.sum()),
        "true_positive": int((actual_high & alerts).sum()),
        "false_positive": int((~actual_high & alerts).sum()),
        "missed_events": int((actual_high & ~alerts).sum()),
        "precision": precision_score(
            actual_high, alerts, zero_division=0
        ),
        "recall": recall_score(
            actual_high, alerts, zero_division=0
        ),
        "f1": f1_score(
            actual_high, alerts, zero_division=0
        ),
    }


def main():

    df = pd.read_csv(
        INPUT,
        parse_dates=[
            "target_start_date",
            "target_end_date",
        ],
    )

    actual = df["actual_cases"].to_numpy(dtype=float)
    predicted = np.maximum(
        df["predicted_cases"].to_numpy(dtype=float),
        0,
    )

    print("=" * 95)
    print("EXPERIMENT D - 2024 OUT-OF-SAMPLE THRESHOLD EVALUATION")
    print("=" * 95)

    results = []

    for threshold in THRESHOLDS:

        metrics = evaluate(
            actual,
            predicted,
            threshold,
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    print("\nOVERALL RESULTS")
    print("-" * 95)

    print(
        results_df.round(4).to_string(index=False)
    )

    # Target month = midpoint of forecast outcome window.
    midpoint = (
        df["target_start_date"]
        + (df["target_end_date"] - df["target_start_date"]) / 2
    )

    df["target_month"] = midpoint.dt.month

    monthly_results = []

    for month, group in df.groupby("target_month"):

        month_actual = group["actual_cases"].to_numpy(dtype=float)
        month_predicted = np.maximum(
            group["predicted_cases"].to_numpy(dtype=float),
            0,
        )

        month_name = pd.Timestamp(
            year=2024,
            month=int(month),
            day=1,
        ).strftime("%B")

        row = {
            "target_month": month_name,
            "records": len(group),
            "actual_high_events": int(
                (month_actual >= ACTUAL_THRESHOLD).sum()
            ),
        }

        for threshold in [27.0, 28.0]:

            metrics = evaluate(
                month_actual,
                month_predicted,
                threshold,
            )

            suffix = str(int(threshold))

            row[f"alerts_{suffix}"] = metrics["alerts"]
            row[f"recall_{suffix}"] = metrics["recall"]
            row[f"precision_{suffix}"] = metrics["precision"]
            row[f"f1_{suffix}"] = metrics["f1"]
            row[f"false_positives_{suffix}"] = metrics["false_positive"]
            row[f"missed_{suffix}"] = metrics["missed_events"]

        monthly_results.append(row)

    monthly_df = pd.DataFrame(monthly_results)

    print("\nMONTHLY COMPARISON: THRESHOLD 27 VS 28")
    print("-" * 95)

    print(
        monthly_df.round(4).to_string(index=False)
    )

    results_df.to_csv(
        OUTPUT,
        index=False,
    )

    print("\nSaved overall comparison:")
    print(OUTPUT)

    print("\nIMPORTANT:")
    print("2024 was used only for evaluation, not threshold selection.")
    print("No model or 2025 test predictions were modified.")


if __name__ == "__main__":
    main()
