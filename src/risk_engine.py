
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent.parent

FEATURES_PATH = ROOT / "data" / "processed" / "dengue_features.csv"
PREDICTIONS_PATH = ROOT / "reports" / "test_predictions.csv"

RISK_OUTPUT_PATH = ROOT / "reports" / "ward_risk_predictions.csv"
THRESHOLD_OUTPUT_PATH = ROOT / "reports" / "risk_thresholds.csv"
REPORT_PATH = ROOT / "reports" / "risk_evaluation.json"

VALIDATION_START = pd.Timestamp("2024-01-01")

RISK_LEVELS = ["Low", "Guarded", "High", "Critical"]


def assign_risk(value, median_threshold, high_threshold):

    if value < 0.5:
        return "Low"

    if value < median_threshold:
        return "Guarded"

    if value < high_threshold:
        return "High"

    return "Critical"


def calculate_thresholds(historical, cities):

    historical = historical[
        historical["target_cases_next_2_4w"] > 0
    ].copy()

    if historical.empty:
        raise ValueError(
            "No positive historical dengue targets are available."
        )

    historical["month"] = historical["date"].dt.month

    monthly_thresholds = (
        historical
        .groupby(["city", "month"])["target_cases_next_2_4w"]
        .quantile([0.50, 0.80])
        .unstack()
        .reset_index()
    )

    monthly_thresholds.columns = [
        "city",
        "month",
        "positive_median",
        "positive_p80"
    ]

    city_thresholds = (
        historical
        .groupby("city")["target_cases_next_2_4w"]
        .quantile([0.50, 0.80])
        .unstack()
    )

    city_thresholds.columns = [
        "city_median",
        "city_p80"
    ]

    national_median = float(
        historical["target_cases_next_2_4w"].quantile(0.50)
    )

    national_p80 = float(
        historical["target_cases_next_2_4w"].quantile(0.80)
    )

    complete_index = pd.MultiIndex.from_product(
        [cities, range(1, 13)],
        names=["city", "month"]
    ).to_frame(index=False)

    thresholds = complete_index.merge(
        monthly_thresholds,
        on=["city", "month"],
        how="left"
    )

    thresholds = thresholds.merge(
        city_thresholds,
        on="city",
        how="left"
    )

    thresholds["threshold_source"] = "city_month"

    missing_monthly = thresholds["positive_median"].isna()

    thresholds.loc[
        missing_monthly, "positive_median"
    ] = thresholds.loc[
        missing_monthly, "city_median"
    ]

    thresholds.loc[
        missing_monthly, "positive_p80"
    ] = thresholds.loc[
        missing_monthly, "city_p80"
    ]

    thresholds.loc[
        missing_monthly, "threshold_source"
    ] = "city_wide_fallback"

    missing_city = thresholds["positive_median"].isna()

    thresholds.loc[
        missing_city, "positive_median"
    ] = national_median

    thresholds.loc[
        missing_city, "positive_p80"
    ] = national_p80

    thresholds.loc[
        missing_city, "threshold_source"
    ] = "national_fallback"

    thresholds = thresholds.drop(
        columns=["city_median", "city_p80"]
    )

    if thresholds[
        ["positive_median", "positive_p80"]
    ].isnull().any().any():
        raise ValueError(
            "Historical thresholds still contain missing values."
        )

    if (
        thresholds["positive_median"]
        > thresholds["positive_p80"]
    ).any():
        raise ValueError(
            "Invalid threshold ordering detected."
        )

    return thresholds


def summarize_tiers(risk_df):

    risk_df = risk_df.copy()

    risk_df["absolute_error"] = (
        risk_df["predicted_cases_next_2_4w"]
        - risk_df["actual_cases_next_2_4w"]
    ).abs()

    summary = (
        risk_df
        .groupby("predicted_risk")
        .agg(
            records=("ward_id", "count"),
            average_predicted_cases=(
                "predicted_cases_next_2_4w", "mean"
            ),
            average_actual_cases=(
                "actual_cases_next_2_4w", "mean"
            ),
            average_absolute_error=("absolute_error", "mean")
        )
        .reindex(RISK_LEVELS)
        .fillna(0)
    )

    return summary


def main():

    print("=" * 65)
    print("DENGUEGUARD AI - REVISED RISK ENGINE")
    print("=" * 65)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}"
        )

    features = pd.read_csv(
        FEATURES_PATH,
        parse_dates=[
            "date",
            "target_start_date",
            "target_end_date"
        ]
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["date"]
    )

    historical = features[
        (features["date"] < VALIDATION_START)
        & (features["target_end_date"] < VALIDATION_START)
    ].copy()

    if historical.empty:
        raise ValueError(
            "No eligible historical training records found."
        )

    cities = sorted(predictions["city"].unique())

    historical_thresholds = calculate_thresholds(
        historical,
        cities
    )

    historical_thresholds.to_csv(
        THRESHOLD_OUTPUT_PATH,
        index=False
    )

    predictions["month"] = predictions["date"].dt.month

    risk_df = predictions.merge(
        historical_thresholds,
        on=["city", "month"],
        how="left",
        validate="many_to_one"
    )

    threshold_columns = [
        "positive_median",
        "positive_p80",
        "threshold_source"
    ]

    if risk_df[threshold_columns].isnull().any().any():
        raise ValueError(
            "Some predictions have no matching risk thresholds."
        )

    risk_df["predicted_risk"] = risk_df.apply(
        lambda row: assign_risk(
            row["predicted_cases_next_2_4w"],
            row["positive_median"],
            row["positive_p80"]
        ),
        axis=1
    )

    risk_df["actual_risk"] = risk_df.apply(
        lambda row: assign_risk(
            row["actual_cases_next_2_4w"],
            row["positive_median"],
            row["positive_p80"]
        ),
        axis=1
    )

    risk_df["risk_match"] = (
        risk_df["predicted_risk"]
        == risk_df["actual_risk"]
    )

    risk_df["absolute_error"] = (
        risk_df["predicted_cases_next_2_4w"]
        - risk_df["actual_cases_next_2_4w"]
    ).abs()

    predicted_summary = summarize_tiers(risk_df)

    actual_summary = (
        risk_df
        .groupby("actual_risk")
        .agg(
            records=("ward_id", "count"),
            average_actual_cases=(
                "actual_cases_next_2_4w", "mean"
            )
        )
        .reindex(RISK_LEVELS)
        .fillna(0)
    )

    city_summary = (
        risk_df
        .groupby(["city", "predicted_risk"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=RISK_LEVELS, fill_value=0)
    )

    monthly_summary = (
        risk_df
        .groupby(["month", "predicted_risk"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=range(1, 13),
            columns=RISK_LEVELS,
            fill_value=0
        )
    )

    confusion = confusion_matrix(
        risk_df["actual_risk"],
        risk_df["predicted_risk"],
        labels=RISK_LEVELS
    )

    classification = classification_report(
        risk_df["actual_risk"],
        risk_df["predicted_risk"],
        labels=RISK_LEVELS,
        output_dict=True,
        zero_division=0
    )

    exact_match_rate = float(
        risk_df["risk_match"].mean()
    )

    threshold_sources = {
        str(key): int(value)
        for key, value in historical_thresholds[
            "threshold_source"
        ].value_counts().items()
    }

    risk_df["date"] = risk_df["date"].dt.strftime(
        "%Y-%m-%d"
    )

    risk_df.to_csv(
        RISK_OUTPUT_PATH,
        index=False
    )

    report = {
        "risk_method": (
            "City-month positive-case percentiles "
            "with city-wide and national fallbacks"
        ),
        "historical_period": "2016-2023",
        "risk_definitions": {
            "Low": "Predicted cases equal zero",
            "Guarded": (
                "Positive prediction below historical "
                "positive-case median"
            ),
            "High": (
                "Positive median to below positive "
                "80th percentile"
            ),
            "Critical": (
                "At or above historical positive "
                "80th percentile"
            )
        },
        "historical_threshold_groups": int(
            len(historical_thresholds)
        ),
        "threshold_source_distribution": threshold_sources,
        "test_records": int(len(risk_df)),
        "exact_risk_tier_match_rate": round(
            exact_match_rate, 4
        ),
        "confusion_matrix_labels": RISK_LEVELS,
        "confusion_matrix": confusion.tolist(),
        "classification_report": classification,
        "predicted_tier_summary": json.loads(
            predicted_summary.round(3).to_json(
                orient="index"
            )
        ),
        "actual_tier_summary": json.loads(
            actual_summary.round(3).to_json(
                orient="index"
            )
        ),
        "city_tier_distribution": json.loads(
            city_summary.to_json(orient="index")
        ),
        "monthly_tier_distribution": json.loads(
            monthly_summary.to_json(orient="index")
        ),
        "threshold_file": str(THRESHOLD_OUTPUT_PATH),
        "risk_predictions_file": str(RISK_OUTPUT_PATH)
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\nHISTORICAL POSITIVE-CASE THRESHOLDS")
    print(historical_thresholds.to_string(index=False))

    print("\nTHRESHOLD SOURCE DISTRIBUTION")
    print(json.dumps(threshold_sources, indent=4))

    print("\nPREDICTED RISK DISTRIBUTION")
    print(predicted_summary.round(3).to_string())

    print("\nACTUAL RISK DISTRIBUTION")
    print(actual_summary.round(3).to_string())

    print("\nCITY-WISE PREDICTED RISK DISTRIBUTION")
    print(city_summary.to_string())

    print("\nMONTH-WISE PREDICTED RISK DISTRIBUTION")
    print(monthly_summary.to_string())

    print("\nRISK TIER EVALUATION")
    print(
        "Exact tier match rate:",
        f"{exact_match_rate:.2%}"
    )

    print("\nCLASSIFICATION REPORT")

    print(
        classification_report(
            risk_df["actual_risk"],
            risk_df["predicted_risk"],
            labels=RISK_LEVELS,
            zero_division=0
        )
    )

    print("\nCONFUSION MATRIX")
    print("Rows = actual risk; columns = predicted risk")

    print(
        pd.DataFrame(
            confusion,
            index=RISK_LEVELS,
            columns=RISK_LEVELS
        )
    )

    print("\nSAVED FILES")
    print("Risk predictions:", RISK_OUTPUT_PATH)
    print("Historical thresholds:", THRESHOLD_OUTPUT_PATH)
    print("Evaluation report:", REPORT_PATH)

    print("\nREVISED RISK ENGINE COMPLETED.")


if __name__ == "__main__":
    main()
