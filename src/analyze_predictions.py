
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = ROOT / "reports" / "test_predictions.csv"
OUTPUT_PATH = ROOT / "reports" / "prediction_diagnostics.json"

def metrics(group):
    actual = group["actual_cases_next_2_4w"]
    predicted = group["predicted_cases_next_2_4w"]

    return {
        "records": int(len(group)),
        "actual_mean": round(float(actual.mean()), 3),
        "predicted_mean": round(float(predicted.mean()), 3),
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
        "bias": round(float((predicted - actual).mean()), 4)
    }

def main():
    df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])

    df["absolute_error"] = (
        df["predicted_cases_next_2_4w"] -
        df["actual_cases_next_2_4w"]
    ).abs()

    df["signed_error"] = (
        df["predicted_cases_next_2_4w"] -
        df["actual_cases_next_2_4w"]
    )

    df["month"] = df["date"].dt.month

    high_threshold = float(
        df["actual_cases_next_2_4w"].quantile(0.90)
    )

    high_cases = df[
        df["actual_cases_next_2_4w"] >= high_threshold
    ]

    city_results = {
        str(city): metrics(group)
        for city, group in df.groupby("city")
    }

    monthly_results = {
        str(month): metrics(group)
        for month, group in df.groupby("month")
    }

    worst_predictions = (
        df.sort_values("absolute_error", ascending=False)
        .head(15)
        [[
            "date",
            "city",
            "ward_id",
            "actual_cases_next_2_4w",
            "predicted_cases_next_2_4w",
            "absolute_error"
        ]]
    )

    report = {
        "overall": metrics(df),
        "city_performance": city_results,
        "monthly_performance": monthly_results,
        "high_incidence_threshold_90th_percentile": round(
            high_threshold, 3
        ),
        "high_incidence_performance": (
            metrics(high_cases) if len(high_cases) else {}
        ),
        "underprediction_records": int(
            (df["signed_error"] < 0).sum()
        ),
        "overprediction_records": int(
            (df["signed_error"] > 0).sum()
        ),
        "worst_predictions": worst_predictions.assign(
            date=worst_predictions["date"].astype(str)
        ).to_dict(orient="records")
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("=" * 65)
    print("DENGUEGUARD AI - PREDICTION DIAGNOSTICS")
    print("=" * 65)

    print("\nOVERALL PERFORMANCE")
    print(json.dumps(report["overall"], indent=4))

    print("\nCITY PERFORMANCE")
    for city, result in city_results.items():
        print(f"\n{city}")
        print(json.dumps(result, indent=4))

    print("\nMONTHLY PERFORMANCE")
    for month, result in monthly_results.items():
        print(f"\nMonth {month}")
        print(json.dumps(result, indent=4))

    print("\nHIGH-INCIDENCE PERFORMANCE")
    print(json.dumps(report["high_incidence_performance"], indent=4))

    print("\nUNDERPREDICTION RECORDS:",
          report["underprediction_records"])
    print("OVERPREDICTION RECORDS:",
          report["overprediction_records"])

    print("\nWORST 15 PREDICTIONS")
    print(worst_predictions.to_string(index=False))

    print("\nReport saved:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
