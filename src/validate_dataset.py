
from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hackathon_preli_dataset.xlsx"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "validation_report.json"
)

SHEET_NAME = "bangladesh_dengue_large_dataset"


def validate_dataset():

    print("=" * 60)
    print("DENGUEGUARD AI - DATA QUALITY VALIDATION")
    print("=" * 60)

    # Load dataset
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    errors = []
    warnings = []
    checks = {}

    # 1. Basic structure
    checks["total_records"] = int(len(df))
    checks["total_columns"] = int(len(df.columns))
    checks["unique_wards"] = int(df["ward_id"].nunique())
    checks["unique_cities"] = int(df["city"].nunique())

    if df.isnull().sum().sum() > 0:
        errors.append("Dataset contains missing values.")

    # 2. Duplicate ward-date records
    duplicate_count = int(
        df.duplicated(subset=["ward_id", "date"]).sum()
    )

    checks["duplicate_ward_date_records"] = duplicate_count

    if duplicate_count > 0:
        errors.append(
            f"Found {duplicate_count} duplicate ward-date records."
        )

    # 3. Records per ward
    ward_counts = df.groupby("ward_id").size()

    checks["minimum_records_per_ward"] = int(ward_counts.min())
    checks["maximum_records_per_ward"] = int(ward_counts.max())

    if not (ward_counts == 522).all():
        errors.append(
            "Not every ward contains exactly 522 records."
        )

    # 4. Weekly date continuity
    weekly_gaps = {}

    for ward, group in df.groupby("ward_id"):
        dates = group["date"].sort_values()
        differences = dates.diff().dropna().dt.days

        invalid_gaps = int((differences != 7).sum())

        if invalid_gaps > 0:
            weekly_gaps[ward] = invalid_gaps

    checks["wards_with_invalid_weekly_gaps"] = len(weekly_gaps)

    if weekly_gaps:
        errors.append(
            "Some wards do not have consistent seven-day intervals."
        )

    # 5. Ward-to-city consistency
    city_counts_per_ward = df.groupby("ward_id")["city"].nunique()

    inconsistent_wards = city_counts_per_ward[
        city_counts_per_ward != 1
    ]

    checks["wards_assigned_to_multiple_cities"] = int(
        len(inconsistent_wards)
    )

    if len(inconsistent_wards) > 0:
        errors.append(
            "Some wards are assigned to multiple cities."
        )

    # 6. Numeric range checks
    range_checks = {
        "dengue_cases_nonnegative": (df["dengue_cases"] >= 0).all(),
        "rainfall_nonnegative": (df["rainfall_mm"] >= 0).all(),
        "population_positive": (
            df["population_density_sqkm"] > 0
        ).all(),
        "humidity_between_0_and_100": (
            df["relative_humidity_pct"].between(0, 100)
        ).all(),
        "water_logging_index_between_0_and_1": (
            df["water_logging_risk_index"].between(0, 1)
        ).all(),
        "sanitation_score_between_0_and_1": (
            df["sanitation_score"].between(0, 1)
        ).all(),
    }

    checks["numeric_range_checks"] = {
        key: bool(value)
        for key, value in range_checks.items()
    }

    for name, passed in range_checks.items():
        if not passed:
            errors.append(f"Failed range check: {name}")

    # 7. Temperature consistency
    invalid_temperature_rows = df[
        (df["min_temp_c"] > df["avg_temp_c"])
        | (df["avg_temp_c"] > df["max_temp_c"])
    ]

    checks["invalid_temperature_rows"] = int(
        len(invalid_temperature_rows)
    )

    if len(invalid_temperature_rows) > 0:
        warnings.append(
            "Some temperature records do not follow "
            "minimum <= average <= maximum."
        )

    # 8. One date range per ward
    ward_date_ranges = df.groupby("ward_id")["date"].agg(
        ["min", "max", "count"]
    )

    checks["overall_start_date"] = str(df["date"].min().date())
    checks["overall_end_date"] = str(df["date"].max().date())

    checks["wards_with_different_date_ranges"] = int(
        (
            (ward_date_ranges["min"] != df["date"].min())
            | (ward_date_ranges["max"] != df["date"].max())
        ).sum()
    )

    if checks["wards_with_different_date_ranges"] > 0:
        warnings.append(
            "Some wards have different date ranges."
        )

    # 9. Check non-finite numeric values
    numeric_df = df.select_dtypes(include="number")

    non_finite_count = int(
        (~np.isfinite(numeric_df.to_numpy(dtype=float))).sum()
    )

    checks["non_finite_numeric_values"] = non_finite_count

    if non_finite_count > 0:
        errors.append(
            "Dataset contains infinite or non-finite numeric values."
        )

    # 10. City distribution
    checks["records_per_city"] = {
        str(city): int(count)
        for city, count in df["city"].value_counts().items()
    }

    # Final report
    report = {
        "status": "PASSED" if not errors else "FAILED",
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "weekly_gap_details": weekly_gaps
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # Display results
    print("\nVALIDATION RESULTS")
    print("-" * 60)

    for key, value in checks.items():
        print(f"{key}: {value}")

    print("\nErrors:", len(errors))

    for error in errors:
        print("ERROR:", error)

    print("\nWarnings:", len(warnings))

    for warning in warnings:
        print("WARNING:", warning)

    print("\nFinal status:", report["status"])
    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    validate_dataset()
