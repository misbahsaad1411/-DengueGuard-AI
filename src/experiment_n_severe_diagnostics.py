
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"

INPUT_PATH = REPORT_DIR / "experiment_l_predictions.csv"
FEATURE_PATH = ROOT / "data" / "processed" / "dengue_features.csv"

SEVERE_THRESHOLD = 45

OUTPUT_SEVERE = REPORT_DIR / "experiment_n_severe_records.csv"
OUTPUT_CITY = REPORT_DIR / "experiment_n_city_analysis.csv"
OUTPUT_MONTH = REPORT_DIR / "experiment_n_month_analysis.csv"
OUTPUT_ERROR = REPORT_DIR / "experiment_n_error_analysis.csv"


# ============================================================
# HELPERS
# ============================================================

def regression_summary(group):

    actual = group["actual"].to_numpy(dtype=float)

    results = {
        "records": len(group),
        "actual_mean": actual.mean(),
        "A_mean": group["model_A"].mean(),
        "Q50_mean": group["Q50"].mean(),
        "A_MAE": np.mean(
            np.abs(actual - group["model_A"])
        ),
        "Q50_MAE": np.mean(
            np.abs(actual - group["Q50"])
        ),
        "A_bias": np.mean(
            group["model_A"] - actual
        ),
        "Q50_bias": np.mean(
            group["Q50"] - actual
        ),
    }

    if "A_50_Q50_50" in group.columns:

        results["blend_mean"] = group["A_50_Q50_50"].mean()

        results["blend_MAE"] = np.mean(
            np.abs(actual - group["A_50_Q50_50"])
        )

        results["blend_bias"] = np.mean(
            group["A_50_Q50_50"] - actual
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 95)
    print("DENGUEGUARD AI - EXPERIMENT N")
    print("SEVERE OUTBREAK DIAGNOSTICS")
    print("=" * 95)

    # --------------------------------------------------------
    # LOAD PREDICTIONS
    # --------------------------------------------------------

    predictions = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"],
    )

    # Load blend predictions.
    blend_path = REPORT_DIR / "experiment_m_predictions.csv"

    blend_columns = [
        "date",
        "city",
        "ward_id",
        "A_50_Q50_50",
    ]

    blend = pd.read_csv(
        blend_path,
        usecols=blend_columns,
        parse_dates=["date"],
    )

    predictions = predictions.merge(
        blend,
        on=["date", "city", "ward_id"],
        how="left",
        validate="one_to_one",
    )

    predictions["month"] = predictions["date"].dt.month

    # --------------------------------------------------------
    # LOAD FEATURE DATA
    # --------------------------------------------------------

    features = pd.read_csv(
        FEATURE_PATH,
        parse_dates=[
            "date",
            "target_start_date",
            "target_end_date",
        ],
    )

    feature_columns = [
        column
        for column in features.columns
        if column not in [
            "target_cases_next_2_4w",
            "target_start_date",
            "target_end_date",
        ]
    ]

    # --------------------------------------------------------
    # MERGE FEATURE INFORMATION
    # --------------------------------------------------------

    diagnostic = predictions.merge(
        features[feature_columns],
        on=["date", "city", "ward_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )

    if diagnostic["actual"].isna().any():
        raise ValueError("Missing actual values after merge.")

    # --------------------------------------------------------
    # IDENTIFY SEVERE RECORDS
    # --------------------------------------------------------

    severe = diagnostic[
        diagnostic["actual"] >= SEVERE_THRESHOLD
    ].copy()

    severe["A_error"] = (
        severe["model_A"] - severe["actual"]
    )

    severe["Q50_error"] = (
        severe["Q50"] - severe["actual"]
    )

    severe["blend_error"] = (
        severe["A_50_Q50_50"] - severe["actual"]
    )

    severe["A_underprediction"] = (
        severe["model_A"] < severe["actual"]
    )

    severe["Q50_underprediction"] = (
        severe["Q50"] < severe["actual"]
    )

    severe["blend_underprediction"] = (
        severe["A_50_Q50_50"] < severe["actual"]
    )

    severe["A_absolute_error"] = severe["A_error"].abs()
    severe["Q50_absolute_error"] = severe["Q50_error"].abs()
    severe["blend_absolute_error"] = severe["blend_error"].abs()

    print("\nSEVERE RECORD SUMMARY")
    print("-" * 95)
    print(f"Severe observations: {len(severe)}")

    print(
        f"Model A underprediction: "
        f"{severe['A_underprediction'].mean() * 100:.2f}%"
    )

    print(
        f"Q50 underprediction: "
        f"{severe['Q50_underprediction'].mean() * 100:.2f}%"
    )

    print(
        f"50/50 blend underprediction: "
        f"{severe['blend_underprediction'].mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CITY ANALYSIS
    # --------------------------------------------------------

    city_results = []

    for city, group in severe.groupby("city"):

        city_results.append({
            "city": city,
            **regression_summary(group),
        })

    city_df = pd.DataFrame(city_results)

    city_df = city_df.sort_values(
        "records",
        ascending=False,
    )

    print("\nSEVERE OUTBREAK BY CITY")
    print("-" * 95)
    print(city_df.to_string(index=False))

    city_df.to_csv(
        OUTPUT_CITY,
        index=False,
    )

    # --------------------------------------------------------
    # MONTH ANALYSIS
    # --------------------------------------------------------

    month_results = []

    for month, group in severe.groupby("month"):

        month_results.append({
            "month": int(month),
            **regression_summary(group),
        })

    month_df = pd.DataFrame(month_results)

    month_df = month_df.sort_values(
        "records",
        ascending=False,
    )

    print("\nSEVERE OUTBREAK BY MONTH")
    print("-" * 95)
    print(month_df.to_string(index=False))

    month_df.to_csv(
        OUTPUT_MONTH,
        index=False,
    )

    # --------------------------------------------------------
    # YEAR ANALYSIS
    # --------------------------------------------------------

    print("\nSEVERE OUTBREAK BY YEAR")
    print("-" * 95)

    year_results = []

    for year, group in severe.groupby("prediction_year"):

        year_results.append({
            "year": int(year),
            **regression_summary(group),
        })

    year_df = pd.DataFrame(year_results)

    print(year_df.to_string(index=False))

    # --------------------------------------------------------
    # ERROR PATTERN ANALYSIS
    # --------------------------------------------------------

    severe["actual_band"] = pd.cut(
        severe["actual"],
        bins=[45, 60, 80, 100, np.inf],
        labels=["45-59", "60-79", "80-99", "100+"],
        right=False,
    )

    error_results = []

    for band, group in severe.groupby(
        "actual_band",
        observed=True,
    ):

        error_results.append({
            "actual_band": str(band),
            **regression_summary(group),
        })

    error_df = pd.DataFrame(error_results)

    print("\nSEVERE INTENSITY BREAKDOWN")
    print("-" * 95)
    print(error_df.to_string(index=False))

    error_df.to_csv(
        OUTPUT_ERROR,
        index=False,
    )

    # --------------------------------------------------------
    # WEATHER AND HISTORY DIAGNOSTICS
    # --------------------------------------------------------

    candidate_features = [
        "avg_temp_c",
        "max_temp_c",
        "min_temp_c",
        "rainfall_mm",
        "relative_humidity_pct",
        "rainfall_mm_lag_1",
        "rainfall_mm_lag_2",
        "rainfall_mm_lag_3",
        "rainfall_mm_lag_4",
        "avg_temp_c_lag_1",
        "avg_temp_c_lag_2",
        "relative_humidity_pct_lag_1",
        "relative_humidity_pct_lag_2",
        "dengue_cases_lag_1",
        "dengue_cases_lag_2",
        "dengue_cases_lag_3",
        "dengue_cases_lag_4",
        "dengue_cases_rolling_mean_4",
        "dengue_cases_rolling_sum_4",
        "rainfall_mm_rolling_mean_4",
        "rainfall_mm_rolling_sum_4",
        "relative_humidity_pct_rolling_mean_4",
    ]

    available = [
        column
        for column in candidate_features
        if column in diagnostic.columns
    ]

    print("\nFEATURE DIAGNOSTICS")
    print("-" * 95)

    feature_comparison = []

    for column in available:

        severe_values = severe[column].dropna()
        normal_values = diagnostic.loc[
            diagnostic["actual"] < SEVERE_THRESHOLD,
            column,
        ].dropna()

        if len(severe_values) == 0 or len(normal_values) == 0:
            continue

        feature_comparison.append({
            "feature": column,
            "severe_mean": severe_values.mean(),
            "normal_mean": normal_values.mean(),
            "severe_median": severe_values.median(),
            "normal_median": normal_values.median(),
            "severe_records": len(severe_values),
            "normal_records": len(normal_values),
        })

    feature_df = pd.DataFrame(feature_comparison)

    if not feature_df.empty:

        feature_df["mean_difference"] = (
            feature_df["severe_mean"]
            - feature_df["normal_mean"]
        )

        print(
            feature_df.sort_values(
                "severe_mean",
                ascending=False,
            ).to_string(index=False)
        )

    # --------------------------------------------------------
    # WORST MISSED SEVERE OBSERVATIONS
    # --------------------------------------------------------

    worst = severe.sort_values(
        "blend_absolute_error",
        ascending=False,
    )

    display_columns = [
        "date",
        "city",
        "ward_id",
        "actual",
        "model_A",
        "Q50",
        "A_50_Q50_50",
        "A_error",
        "Q50_error",
        "blend_error",
    ]

    print("\nTOP 25 WORST BLEND ERRORS")
    print("-" * 95)

    print(
        worst[display_columns]
        .head(25)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # SAVE SEVERE RECORDS
    # --------------------------------------------------------

    severe.to_csv(
        OUTPUT_SEVERE,
        index=False,
    )

    print("\nSAVED REPORTS")
    print(OUTPUT_SEVERE)
    print(OUTPUT_CITY)
    print(OUTPUT_MONTH)
    print(OUTPUT_ERROR)

    print("\nEXPERIMENT N COMPLETE")


if __name__ == "__main__":
    main()
