
from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hackathon_preli_dataset.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dengue_features.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "feature_engineering_report.json"

SHEET_NAME = "bangladesh_dengue_large_dataset"

WEATHER_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "relative_humidity_pct"
]

STATIC_COLUMNS = [
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score"
]


def create_features(df):

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["ward_id", "date"]
    ).reset_index(drop=True)

    grouped = df.groupby("ward_id", sort=False)

    for column in WEATHER_COLUMNS:
        for lag in range(1, 5):
            df[f"{column}_lag_{lag}"] = grouped[column].shift(lag)

    for lag in range(1, 5):
        df[f"dengue_cases_lag_{lag}"] = grouped["dengue_cases"].shift(lag)

    for column in WEATHER_COLUMNS:
        df[f"{column}_rolling_4w"] = grouped[column].transform(
            lambda values: values.shift(1).rolling(
                window=4,
                min_periods=4
            ).mean()
        )

    df["rainfall_rolling_4w_sum"] = grouped["rainfall_mm"].transform(
        lambda values: values.shift(1).rolling(
            window=4,
            min_periods=4
        ).sum()
    )

    df["dengue_cases_rolling_4w"] = grouped["dengue_cases"].transform(
        lambda values: values.shift(1).rolling(
            window=4,
            min_periods=4
        ).mean()
    )

    df["dengue_cases_rolling_4w_sum"] = grouped["dengue_cases"].transform(
        lambda values: values.shift(1).rolling(
            window=4,
            min_periods=4
        ).sum()
    )

    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    df["week_sin"] = np.sin(
        2 * np.pi * df["week_of_year"] / 52.1775
    )

    df["week_cos"] = np.cos(
        2 * np.pi * df["week_of_year"] / 52.1775
    )

    future_2 = grouped["dengue_cases"].shift(-2)
    future_3 = grouped["dengue_cases"].shift(-3)
    future_4 = grouped["dengue_cases"].shift(-4)

    df["target_cases_next_2_4w"] = (
        future_2 + future_3 + future_4
    )

    df["target_start_date"] = grouped["date"].shift(-2)
    df["target_end_date"] = grouped["date"].shift(-4)

    df = df.dropna().reset_index(drop=True)

    df["target_cases_next_2_4w"] = (
        df["target_cases_next_2_4w"].astype("int64")
    )

    return df


def main():

    print("=" * 60)
    print("DENGUEGUARD AI - FEATURE ENGINEERING")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_excel(
        DATA_PATH,
        sheet_name=SHEET_NAME
    )

    required_columns = [
        "date",
        "city",
        "ward_id",
        "dengue_cases",
        *WEATHER_COLUMNS,
        *STATIC_COLUMNS
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print(f"\nOriginal records: {len(df)}")
    print(f"Original wards: {df['ward_id'].nunique()}")

    features = create_features(df)

    feature_columns = [
        column
        for column in features.columns
        if column not in [
            "date",
            "dengue_cases",
            "target_cases_next_2_4w",
            "target_start_date",
            "target_end_date"
        ]
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False
    )

    report = {
        "original_records": int(len(df)),
        "engineered_records": int(len(features)),
        "removed_records": int(len(df) - len(features)),
        "unique_wards": int(features["ward_id"].nunique()),
        "unique_cities": int(features["city"].nunique()),
        "feature_count": int(len(feature_columns)),
        "target_column": "target_cases_next_2_4w",
        "target_definition": "Sum of dengue cases at weeks t+2, t+3 and t+4",
        "start_date": str(features["date"].min().date()),
        "end_date": str(features["date"].max().date()),
        "missing_values": int(features.isnull().sum().sum()),
        "target_minimum": int(features["target_cases_next_2_4w"].min()),
        "target_maximum": int(features["target_cases_next_2_4w"].max()),
        "target_mean": round(
            float(features["target_cases_next_2_4w"].mean()), 3
        ),
        "output_file": str(OUTPUT_PATH)
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\nFEATURE ENGINEERING RESULTS")
    print("-" * 60)

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nGenerated feature columns:")
    for column in feature_columns:
        print("-", column)

    print("\nFirst 5 engineered records:")
    print(features.head().to_string())

    print("\nFeature engineering completed successfully.")


if __name__ == "__main__":
    main()
