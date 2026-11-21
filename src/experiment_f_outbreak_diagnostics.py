
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

INPUT_2023 = REPORTS / "experiment_d_2023_calibration.csv"
INPUT_2024 = REPORTS / "validation_predictions.csv"

OUTPUT_SEVERITY = REPORTS / "experiment_f_severity.csv"
OUTPUT_MONTHLY = REPORTS / "experiment_f_monthly.csv"
OUTPUT_CITY = REPORTS / "experiment_f_city.csv"
OUTPUT_CITY_MONTH = REPORTS / "experiment_f_city_month.csv"
OUTPUT_JSON = REPORTS / "experiment_f_diagnostics.json"

HIGH_THRESHOLD = 28.0
PREDICTION_ALERT_THRESHOLD = 28.0


def load_predictions(path, period):

    df = pd.read_csv(
        path,
        parse_dates=[
            "target_start_date",
            "target_end_date",
        ],
    )

    required = [
        "target_start_date",
        "target_end_date",
        "city",
        "ward_id",
        "actual_cases",
        "predicted_cases",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {missing}"
        )

    df["actual_cases"] = pd.to_numeric(
        df["actual_cases"],
        errors="raise",
    )

    df["predicted_cases"] = pd.to_numeric(
        df["predicted_cases"],
        errors="raise",
    ).clip(lower=0)

    midpoint = (
        df["target_start_date"]
        + (
            df["target_end_date"]
            - df["target_start_date"]
        ) / 2
    )

    df["target_month"] = midpoint.dt.month

    df["period"] = period

    df["error"] = (
        df["predicted_cases"] - df["actual_cases"]
    )

    df["absolute_error"] = df["error"].abs()

    df["actual_high"] = (
        df["actual_cases"] >= HIGH_THRESHOLD
    )

    df["predicted_high"] = (
        df["predicted_cases"] >= PREDICTION_ALERT_THRESHOLD
    )

    df["severity"] = pd.cut(
        df["actual_cases"],
        bins=[-np.inf, 0.9999, 9.9999, 27.9999, np.inf],
        labels=[
            "Zero",
            "Low (1-9)",
            "Moderate (10-27)",
            "High (28+)",
        ],
    )

    return df


def summarize(group):

    actual = group["actual_cases"].to_numpy(dtype=float)
    predicted = group["predicted_cases"].to_numpy(dtype=float)

    error = predicted - actual

    high_mask = actual >= HIGH_THRESHOLD

    high_count = int(high_mask.sum())

    true_positive = int(
        (
            high_mask
            & (predicted >= PREDICTION_ALERT_THRESHOLD)
        ).sum()
    )

    false_positive = int(
        (
            ~high_mask
            & (predicted >= PREDICTION_ALERT_THRESHOLD)
        ).sum()
    )

    missed = high_count - true_positive

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / high_count
        if high_count > 0
        else None
    )

    return {
        "records": int(len(group)),
        "actual_mean": float(np.mean(actual)),
        "predicted_mean": float(np.mean(predicted)),
        "bias": float(np.mean(error)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "actual_max": float(np.max(actual)),
        "predicted_max": float(np.max(predicted)),
        "actual_p90": float(np.percentile(actual, 90)),
        "predicted_p90": float(np.percentile(predicted, 90)),
        "underprediction_rate": float(np.mean(error < 0)),
        "overprediction_rate": float(np.mean(error > 0)),
        "actual_high_events": high_count,
        "detected_high_events": true_positive,
        "missed_high_events": missed,
        "false_positive_alerts": false_positive,
        "high_event_precision": float(precision),
        "high_event_recall": (
            float(recall) if recall is not None else None
        ),
        "high_event_actual_mean": (
            float(np.mean(actual[high_mask]))
            if high_count else None
        ),
        "high_event_predicted_mean": (
            float(np.mean(predicted[high_mask]))
            if high_count else None
        ),
        "high_event_mae": (
            float(mean_absolute_error(
                actual[high_mask],
                predicted[high_mask],
            ))
            if high_count else None
        ),
        "high_event_bias": (
            float(np.mean(error[high_mask]))
            if high_count else None
        ),
    }


def month_name(month):

    return pd.Timestamp(
        year=2024,
        month=int(month),
        day=1,
    ).strftime("%B")


def main():

    print("=" * 95)
    print("DENGUEGUARD AI - EXPERIMENT F")
    print("OUTBREAK INTENSITY DIAGNOSTICS")
    print("=" * 95)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    data_2023 = load_predictions(
        INPUT_2023,
        "2023 calibration",
    )

    data_2024 = load_predictions(
        INPUT_2024,
        "2024 validation",
    )

    combined = pd.concat(
        [data_2023, data_2024],
        ignore_index=True,
    )

    print("\nDATA SUMMARY")
    print("-" * 95)

    for period, group in combined.groupby("period"):

        print(
            f"{period}: {len(group)} records, "
            f"{group['actual_high'].sum()} actual high events"
        )

    # --------------------------------------------------------
    # 2. SEVERITY ANALYSIS
    # --------------------------------------------------------

    severity_results = []

    for (period, severity), group in combined.groupby(
        ["period", "severity"],
        observed=True,
    ):

        metrics = summarize(group)

        severity_results.append({
            "period": period,
            "severity": str(severity),
            **metrics,
        })

    severity_df = pd.DataFrame(severity_results)

    print("\nACTUAL SEVERITY ANALYSIS")
    print("-" * 95)

    severity_columns = [
        "period",
        "severity",
        "records",
        "actual_mean",
        "predicted_mean",
        "bias",
        "mae",
        "underprediction_rate",
    ]

    print(
        severity_df[severity_columns]
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 3. TARGET MONTH ANALYSIS
    # --------------------------------------------------------

    monthly_results = []

    for (period, month), group in combined.groupby(
        ["period", "target_month"]
    ):

        metrics = summarize(group)

        monthly_results.append({
            "period": period,
            "target_month": int(month),
            "month_name": month_name(month),
            **metrics,
        })

    monthly_df = pd.DataFrame(monthly_results)

    print("\nMONTHLY OUTBREAK PERFORMANCE")
    print("-" * 95)

    monthly_columns = [
        "period",
        "month_name",
        "records",
        "actual_high_events",
        "actual_mean",
        "predicted_mean",
        "bias",
        "mae",
        "high_event_mae",
        "high_event_recall",
        "missed_high_events",
        "false_positive_alerts",
    ]

    print(
        monthly_df[monthly_columns]
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 4. CITY ANALYSIS
    # --------------------------------------------------------

    city_results = []

    for (period, city), group in combined.groupby(
        ["period", "city"]
    ):

        metrics = summarize(group)

        city_results.append({
            "period": period,
            "city": city,
            **metrics,
        })

    city_df = pd.DataFrame(city_results)

    print("\nCITY-LEVEL OUTBREAK PERFORMANCE")
    print("-" * 95)

    city_columns = [
        "period",
        "city",
        "records",
        "actual_high_events",
        "bias",
        "mae",
        "high_event_mae",
        "high_event_recall",
        "missed_high_events",
        "false_positive_alerts",
    ]

    print(
        city_df[city_columns]
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 5. CITY x TARGET MONTH ANALYSIS
    # --------------------------------------------------------

    city_month_results = []

    for (period, city, month), group in combined.groupby(
        ["period", "city", "target_month"]
    ):

        metrics = summarize(group)

        city_month_results.append({
            "period": period,
            "city": city,
            "target_month": int(month),
            "month_name": month_name(month),
            **metrics,
        })

    city_month_df = pd.DataFrame(city_month_results)

    # Focus on July, August and September.
    outbreak_months = city_month_df[
        city_month_df["target_month"].isin([7, 8, 9])
    ].copy()

    print("\nCITY x OUTBREAK-MONTH ANALYSIS")
    print("-" * 95)

    print(
        outbreak_months[
            [
                "period",
                "city",
                "month_name",
                "records",
                "actual_high_events",
                "bias",
                "mae",
                "high_event_mae",
                "high_event_recall",
                "missed_high_events",
                "false_positive_alerts",
            ]
        ].round(4).to_string(index=False)
    )

    # --------------------------------------------------------
    # 6. SAVE OUTPUTS
    # --------------------------------------------------------

    severity_df.to_csv(
        OUTPUT_SEVERITY,
        index=False,
    )

    monthly_df.to_csv(
        OUTPUT_MONTHLY,
        index=False,
    )

    city_df.to_csv(
        OUTPUT_CITY,
        index=False,
    )

    city_month_df.to_csv(
        OUTPUT_CITY_MONTH,
        index=False,
    )

    # Overall summaries for JSON.
    overall_results = {}

    for period, group in combined.groupby("period"):

        overall_results[period] = summarize(group)

    report = {
        "experiment": "F - Outbreak intensity diagnostics",
        "actual_high_threshold": HIGH_THRESHOLD,
        "prediction_alert_threshold": PREDICTION_ALERT_THRESHOLD,
        "periods": {
            "2023": "Calibration predictions",
            "2024": "Experiment A validation predictions",
        },
        "overall": overall_results,
        "severity_results": severity_results,
        "monthly_results": monthly_results,
        "city_results": city_results,
        "city_month_results": city_month_results,
        "model_modified": False,
        "risk_thresholds_modified": False,
        "2025_test_used": False,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print("\nSAVED REPORTS")
    print("-" * 95)

    for path in [
        OUTPUT_SEVERITY,
        OUTPUT_MONTHLY,
        OUTPUT_CITY,
        OUTPUT_CITY_MONTH,
        OUTPUT_JSON,
    ]:
        print(path)

    print("\nEXPERIMENT F COMPLETED.")
    print("No model, risk thresholds or test predictions were changed.")


if __name__ == "__main__":
    main()
