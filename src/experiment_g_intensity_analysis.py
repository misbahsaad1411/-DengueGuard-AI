
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

INPUTS = {
    "2023 calibration": REPORTS / "experiment_d_2023_calibration.csv",
    "2024 validation": REPORTS / "validation_predictions.csv",
}

OUTPUT_CSV = REPORTS / "experiment_g_intensity.csv"
OUTPUT_JSON = REPORTS / "experiment_g_diagnostics.json"

ALERT_THRESHOLD = 28.0

BINS = [-np.inf, 27.9999, 34.9999, 44.9999, 59.9999, np.inf]
LABELS = [
    "Below 28",
    "28-34",
    "35-44",
    "45-59",
    "60+",
]


def load_data(path, period):

    df = pd.read_csv(path)

    required = [
        "actual_cases",
        "predicted_cases",
        "city",
        "ward_id",
        "target_start_date",
        "target_end_date",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df["actual_cases"] = pd.to_numeric(
        df["actual_cases"], errors="raise"
    )

    df["predicted_cases"] = pd.to_numeric(
        df["predicted_cases"], errors="raise"
    ).clip(lower=0)

    df["error"] = (
        df["predicted_cases"] - df["actual_cases"]
    )

    df["absolute_error"] = df["error"].abs()

    df["actual_intensity"] = pd.cut(
        df["actual_cases"],
        bins=BINS,
        labels=LABELS,
    )

    df["actual_high"] = df["actual_cases"] >= ALERT_THRESHOLD
    df["predicted_high"] = df["predicted_cases"] >= ALERT_THRESHOLD

    df["period"] = period

    return df


def summarize(group):

    actual = group["actual_cases"].to_numpy(dtype=float)
    predicted = group["predicted_cases"].to_numpy(dtype=float)
    error = predicted - actual

    high = actual >= ALERT_THRESHOLD

    detected = (
        high & (predicted >= ALERT_THRESHOLD)
    )

    missed = (
        high & (predicted < ALERT_THRESHOLD)
    )

    false_positive = (
        ~high & (predicted >= ALERT_THRESHOLD)
    )

    high_count = int(high.sum())
    detected_count = int(detected.sum())
    false_count = int(false_positive.sum())

    return {
        "records": int(len(group)),
        "actual_mean": float(np.mean(actual)),
        "predicted_mean": float(np.mean(predicted)),
        "actual_max": float(np.max(actual)),
        "predicted_max": float(np.max(predicted)),
        "bias": float(np.mean(error)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "underprediction_rate": float(np.mean(error < 0)),
        "overprediction_rate": float(np.mean(error > 0)),
        "actual_high_events": high_count,
        "detected_high_events": detected_count,
        "missed_high_events": int(missed.sum()),
        "false_positive_alerts": false_count,
        "high_event_recall": (
            float(detected_count / high_count)
            if high_count else None
        ),
        "high_event_precision": (
            float(detected_count / (detected_count + false_count))
            if detected_count + false_count else None
        ),
    }


def main():

    print("=" * 95)
    print("DENGUEGUARD AI - EXPERIMENT G")
    print("HIGH-INTENSITY ERROR ANALYSIS")
    print("=" * 95)

    frames = []

    for period, path in INPUTS.items():

        if not path.exists():
            raise FileNotFoundError(path)

        df = load_data(path, period)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # --------------------------------------------------
    # 1. INTENSITY GROUP ANALYSIS
    # --------------------------------------------------

    results = []

    for (period, intensity), group in combined.groupby(
        ["period", "actual_intensity"],
        observed=True,
    ):

        results.append({
            "period": period,
            "intensity": str(intensity),
            **summarize(group),
        })

    result_df = pd.DataFrame(results)

    print("\nINTENSITY PERFORMANCE")
    print("-" * 95)

    columns = [
        "period",
        "intensity",
        "records",
        "actual_mean",
        "predicted_mean",
        "bias",
        "mae",
        "underprediction_rate",
        "actual_high_events",
        "missed_high_events",
        "high_event_recall",
    ]

    print(
        result_df[columns]
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # 2. EXTREME EVENT ANALYSIS
    # --------------------------------------------------

    print("\nEXTREME EVENT PERFORMANCE")
    print("-" * 95)

    extreme = combined[
        combined["actual_cases"] >= 45
    ]

    for period, group in extreme.groupby("period"):

        if group.empty:
            continue

        metrics = summarize(group)

        print(f"\n{period}")

        for key, value in metrics.items():
            print(f"{key}: {value}")

    # --------------------------------------------------
    # 3. CITY-LEVEL EXTREME EVENTS
    # --------------------------------------------------

    city_results = []

    for (period, city), group in extreme.groupby(
        ["period", "city"]
    ):

        city_results.append({
            "period": period,
            "city": city,
            **summarize(group),
        })

    city_df = pd.DataFrame(city_results)

    print("\nCITY-LEVEL EXTREME EVENT PERFORMANCE")
    print("-" * 95)

    if not city_df.empty:

        print(
            city_df[
                [
                    "period",
                    "city",
                    "records",
                    "actual_mean",
                    "predicted_mean",
                    "bias",
                    "mae",
                    "underprediction_rate",
                    "missed_high_events",
                ]
            ].round(4).to_string(index=False)
        )

    # --------------------------------------------------
    # 4. LARGEST UNDERPREDICTIONS
    # --------------------------------------------------

    print("\nLARGEST UNDERPREDICTIONS")
    print("-" * 95)

    worst = combined.sort_values(
        "error",
        ascending=True,
    ).head(20)

    worst_columns = [
        "period",
        "city",
        "ward_id",
        "target_start_date",
        "target_end_date",
        "actual_cases",
        "predicted_cases",
        "error",
    ]

    print(
        worst[worst_columns].round(3).to_string(index=False)
    )

    # --------------------------------------------------
    # 5. SAVE REPORTS
    # --------------------------------------------------

    result_df.to_csv(OUTPUT_CSV, index=False)

    report = {
        "experiment": "G - High-intensity error analysis",
        "alert_threshold": ALERT_THRESHOLD,
        "intensity_results": results,
        "extreme_event_results": {
            period: summarize(group)
            for period, group in extreme.groupby("period")
        },
        "city_extreme_results": (
            city_results
        ),
        "largest_underpredictions": (
            worst[worst_columns]
            .assign(
                target_start_date=lambda x: x["target_start_date"].astype(str),
                target_end_date=lambda x: x["target_end_date"].astype(str),
            )
            .to_dict(orient="records")
        ),
        "model_modified": False,
        "risk_thresholds_modified": False,
        "2025_test_used": False,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nSAVED:")
    print(OUTPUT_CSV)
    print(OUTPUT_JSON)

    print("\nEXPERIMENT G COMPLETED.")
    print("No model or test predictions were changed.")


if __name__ == "__main__":
    main()
