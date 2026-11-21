
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

CALIBRATION_PATH = REPORTS / "experiment_d_2023_calibration.csv"
VALIDATION_PATH = REPORTS / "validation_predictions.csv"

OUTPUT_MONTHLY = REPORTS / "experiment_e_monthly_results.csv"
OUTPUT_PREDICTIONS = REPORTS / "experiment_e_predictions.csv"
OUTPUT_JSON = REPORTS / "experiment_e_diagnostics.json"

HIGH_THRESHOLD = 28.0

# Predefined shrinkage strength.
# Larger values shrink monthly corrections more strongly
# toward the overall calibration correction.
SHRINKAGE_STRENGTH = 60.0


def add_target_month(df):

    df = df.copy()

    midpoint = (
        df["target_start_date"]
        + (
            df["target_end_date"]
            - df["target_start_date"]
        ) / 2
    )

    df["target_month"] = midpoint.dt.month

    return df


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float),
        0,
    )

    errors = predicted - actual
    high_mask = actual >= HIGH_THRESHOLD

    high_mae = (
        mean_absolute_error(
            actual[high_mask],
            predicted[high_mask],
        )
        if high_mask.any()
        else None
    )

    return {
        "records": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "bias": float(np.mean(errors)),
        "actual_mean": float(np.mean(actual)),
        "predicted_mean": float(np.mean(predicted)),
        "high_event_records": int(high_mask.sum()),
        "high_event_mae": (
            float(high_mae)
            if high_mae is not None
            else None
        ),
        "overpredictions": int(np.sum(errors > 0)),
        "underpredictions": int(np.sum(errors < 0)),
    }


def main():

    print("=" * 90)
    print("DENGUEGUARD AI - EXPERIMENT E")
    print("SEASONAL FORECAST BIAS CORRECTION")
    print("=" * 90)

    # --------------------------------------------------------
    # 1. LOAD 2023 CALIBRATION PREDICTIONS
    # --------------------------------------------------------

    calibration = pd.read_csv(
        CALIBRATION_PATH,
        parse_dates=[
            "target_start_date",
            "target_end_date",
        ],
    )

    required_calibration = [
        "target_start_date",
        "target_end_date",
        "actual_cases",
        "predicted_cases",
    ]

    for column in required_calibration:
        if column not in calibration.columns:
            raise ValueError(
                f"Missing calibration column: {column}"
            )

    calibration = add_target_month(calibration)

    calibration["residual"] = (
        calibration["actual_cases"]
        - calibration["predicted_cases"]
    )

    # Positive residual = model underpredicted.
    # Negative residual = model overpredicted.

    global_correction = float(
        calibration["residual"].mean()
    )

    monthly_corrections = (
        calibration.groupby("target_month")["residual"]
        .agg(["mean", "count"])
        .reset_index()
    )

    monthly_corrections["shrinkage_weight"] = (
        monthly_corrections["count"]
        / (
            monthly_corrections["count"]
            + SHRINKAGE_STRENGTH
        )
    )

    monthly_corrections["correction"] = (
        global_correction
        + monthly_corrections["shrinkage_weight"]
        * (
            monthly_corrections["mean"]
            - global_correction
        )
    )

    correction_map = dict(
        zip(
            monthly_corrections["target_month"],
            monthly_corrections["correction"],
        )
    )

    print("\n2023 CALIBRATION DATA")
    print("-" * 90)
    print("Records:", len(calibration))
    print("Global correction:", round(global_correction, 4))
    print("Shrinkage strength:", SHRINKAGE_STRENGTH)

    print("\nMONTHLY CORRECTIONS LEARNED FROM 2023")
    print("-" * 90)

    print(
        monthly_corrections.round(4).to_string(index=False)
    )

    # --------------------------------------------------------
    # 2. LOAD ORIGINAL EXPERIMENT A VALIDATION PREDICTIONS
    # --------------------------------------------------------

    validation = pd.read_csv(
        VALIDATION_PATH,
        parse_dates=[
            "date",
            "target_start_date",
            "target_end_date",
        ],
    )

    required_validation = [
        "date",
        "target_start_date",
        "target_end_date",
        "city",
        "ward_id",
        "actual_cases",
        "predicted_cases",
    ]

    for column in required_validation:
        if column not in validation.columns:
            raise ValueError(
                f"Missing validation column: {column}"
            )

    validation = validation.sort_values(
        ["date", "ward_id"]
    ).reset_index(drop=True)

    validation = add_target_month(validation)

    if not (
        (validation["date"] >= "2024-01-01")
        & (validation["date"] < "2025-01-01")
        & (validation["target_end_date"] < "2025-01-01")
    ).all():
        raise ValueError(
            "Validation file contains records outside the expected period."
        )

    actual = validation["actual_cases"].to_numpy(dtype=float)

    original_predictions = np.maximum(
        validation["predicted_cases"].to_numpy(dtype=float),
        0,
    )

    # --------------------------------------------------------
    # 3. APPLY TWO CORRECTIONS
    # --------------------------------------------------------

    global_predictions = np.maximum(
        original_predictions + global_correction,
        0,
    )

    monthly_adjustments = (
        validation["target_month"]
        .map(correction_map)
        .fillna(global_correction)
        .to_numpy(dtype=float)
    )

    seasonal_predictions = np.maximum(
        original_predictions + monthly_adjustments,
        0,
    )

    # --------------------------------------------------------
    # 4. OVERALL METRICS
    # --------------------------------------------------------

    original_metrics = calculate_metrics(
        actual,
        original_predictions,
    )

    global_metrics = calculate_metrics(
        actual,
        global_predictions,
    )

    seasonal_metrics = calculate_metrics(
        actual,
        seasonal_predictions,
    )

    comparison = pd.DataFrame([
        {
            "method": "Original Experiment A",
            **original_metrics,
        },
        {
            "method": "Global bias correction",
            **global_metrics,
        },
        {
            "method": "Seasonal shrunk correction",
            **seasonal_metrics,
        },
    ])

    print("\n2024 OVERALL PERFORMANCE")
    print("-" * 90)

    print(
        comparison.round(4).to_string(index=False)
    )

    # --------------------------------------------------------
    # 5. MONTHLY METRICS
    # --------------------------------------------------------

    monthly_results = []

    for month, group in validation.groupby("target_month"):

        indices = group.index.to_numpy()

        month_actual = actual[indices]
        month_original = original_predictions[indices]
        month_global = global_predictions[indices]
        month_seasonal = seasonal_predictions[indices]

        original = calculate_metrics(
            month_actual,
            month_original,
        )

        seasonal = calculate_metrics(
            month_actual,
            month_seasonal,
        )

        month_name = pd.Timestamp(
            year=2024,
            month=int(month),
            day=1,
        ).strftime("%B")

        monthly_results.append({
            "target_month": month_name,
            "records": len(group),
            "actual_high_events": original["high_event_records"],
            "original_mae": original["mae"],
            "seasonal_mae": seasonal["mae"],
            "original_rmse": original["rmse"],
            "seasonal_rmse": seasonal["rmse"],
            "original_bias": original["bias"],
            "seasonal_bias": seasonal["bias"],
            "original_high_event_mae": original["high_event_mae"],
            "seasonal_high_event_mae": seasonal["high_event_mae"],
        })

    monthly_df = pd.DataFrame(monthly_results)

    print("\n2024 TARGET-MONTH PERFORMANCE")
    print("-" * 90)

    print(
        monthly_df.round(4).to_string(index=False)
    )

    # --------------------------------------------------------
    # 6. SAVE PREDICTIONS
    # --------------------------------------------------------

    output = validation[
        [
            "date",
            "target_start_date",
            "target_end_date",
            "city",
            "ward_id",
            "actual_cases",
        ]
    ].copy()

    output["original_prediction"] = original_predictions
    output["global_corrected_prediction"] = global_predictions
    output["seasonal_corrected_prediction"] = seasonal_predictions
    output["seasonal_correction_applied"] = monthly_adjustments

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    monthly_df.to_csv(
        OUTPUT_MONTHLY,
        index=False,
    )

    # --------------------------------------------------------
    # 7. SAVE DIAGNOSTICS
    # --------------------------------------------------------

    report = {
        "experiment": "E - Seasonal forecast bias correction",
        "calibration_source": "2023 out-of-time predictions",
        "evaluation_source": "2024 Experiment A validation predictions",
        "global_correction": global_correction,
        "shrinkage_strength": SHRINKAGE_STRENGTH,
        "monthly_corrections": monthly_corrections.to_dict(
            orient="records"
        ),
        "overall_results": {
            "original": original_metrics,
            "global_corrected": global_metrics,
            "seasonal_corrected": seasonal_metrics,
        },
        "monthly_results": monthly_results,
        "model_modified": False,
        "2025_test_used": False,
        "2025_test_modified": False,
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

    print("\nSAVED FILES")
    print("-" * 90)
    print(OUTPUT_PREDICTIONS)
    print(OUTPUT_MONTHLY)
    print(OUTPUT_JSON)

    print("\nEXPERIMENT E COMPLETED.")
    print("Original model and 2025 test predictions were not modified.")


if __name__ == "__main__":
    main()
