
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

a = pd.read_csv(
    REPORTS / "validation_predictions.csv",
    parse_dates=["date", "target_start_date", "target_end_date"],
)

b = pd.read_csv(
    REPORTS / "experiment_b_predictions.csv",
    parse_dates=["date", "target_start_date", "target_end_date"],
)

keys = ["date", "city", "ward_id"]

a = a.sort_values(keys).reset_index(drop=True)
b = b.sort_values(keys).reset_index(drop=True)

if not a[keys].equals(b[keys]):
    raise ValueError("Experiment A and B rows do not match.")

if not np.allclose(a["actual_cases"], b["actual_cases"]):
    raise ValueError("Actual target values do not match.")

if not a["target_start_date"].equals(b["target_start_date"]):
    raise ValueError("Target start dates do not match.")

if not a["target_end_date"].equals(b["target_end_date"]):
    raise ValueError("Target end dates do not match.")

# Target window is t+2 through t+4.
# Use its middle week to assign a seasonal month.
midpoint = (
    a["target_start_date"]
    + (a["target_end_date"] - a["target_start_date"]) / 2
)

a["target_month"] = midpoint.dt.month

models = {
    "A_original": a["predicted_cases"].to_numpy(dtype=float),
    "B_weight_2x": b["prediction_weight_2_0"].to_numpy(dtype=float),
    "B_weight_3x": b["prediction_weight_3_0"].to_numpy(dtype=float),
}

HIGH_THRESHOLD = 28.0
OUTBREAK_MONTHS = [7, 8, 9]

results = []

for month in OUTBREAK_MONTHS:

    for city in sorted(a["city"].unique()):

        mask = (
            (a["target_month"] == month)
            & (a["city"] == city)
        )

        group = a.loc[mask]
        actual = group["actual_cases"].to_numpy(dtype=float)

        if len(group) == 0:
            continue

        actual_high = actual >= HIGH_THRESHOLD
        event_count = int(actual_high.sum())

        for model_name, all_predictions in models.items():

            predicted = np.maximum(
                all_predictions[group.index.to_numpy()],
                0,
            )

            predicted_high = predicted >= HIGH_THRESHOLD

            tp = int(np.sum(actual_high & predicted_high))
            fp = int(np.sum(~actual_high & predicted_high))
            fn = int(np.sum(actual_high & ~predicted_high))

            precision = (
                tp / (tp + fp)
                if tp + fp else np.nan
            )

            recall = (
                tp / (tp + fn)
                if tp + fn else np.nan
            )

            f1 = (
                2 * precision * recall / (precision + recall)
                if not np.isnan(precision)
                and not np.isnan(recall)
                and precision + recall > 0
                else np.nan
            )

            results.append({
                "target_month": pd.Timestamp(
                    year=2024, month=month, day=1
                ).strftime("%B"),
                "city": city,
                "model": model_name,
                "records": len(group),
                "actual_high_events": event_count,
                "actual_mean": round(float(actual.mean()), 3),
                "mae": round(
                    float(mean_absolute_error(actual, predicted)), 4
                ),
                "bias": round(float(np.mean(predicted - actual)), 4),
                "true_positive": tp,
                "false_positive": fp,
                "missed_events": fn,
                "precision": (
                    round(float(precision), 4)
                    if not np.isnan(precision) else None
                ),
                "recall": (
                    round(float(recall), 4)
                    if not np.isnan(recall) else None
                ),
                "f1": (
                    round(float(f1), 4)
                    if not np.isnan(f1) else None
                ),
            })

result_df = pd.DataFrame(results)

output_path = REPORTS / "outbreak_city_comparison.csv"
result_df.to_csv(output_path, index=False)

print("=" * 110)
print("DENGUEGUARD AI - OUTBREAK PERFORMANCE BY CITY")
print("=" * 110)

for month in ["July", "August", "September"]:

    month_df = result_df[
        result_df["target_month"] == month
    ]

    print(f"\n{month.upper()}")
    print("-" * 110)

    print(
        month_df[
            [
                "city",
                "model",
                "records",
                "actual_high_events",
                "mae",
                "bias",
                "true_positive",
                "false_positive",
                "missed_events",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )

print("\nSAVED:", output_path)
print("Analysis completed. No model or test predictions were modified.")
