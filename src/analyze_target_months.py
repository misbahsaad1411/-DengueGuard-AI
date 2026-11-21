
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# DENGUEGUARD AI - TARGET MONTH DIAGNOSTICS
# Compare Experiments A, B and C on identical validation rows.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"

A_PATH = REPORT_DIR / "validation_predictions.csv"
B_PATH = REPORT_DIR / "experiment_b_predictions.csv"
C_PATH = REPORT_DIR / "experiment_c_predictions.csv"

OUTPUT_CSV = REPORT_DIR / "target_month_comparison.csv"
OUTPUT_JSON = REPORT_DIR / "target_month_comparison.json"

HIGH_THRESHOLD = 28.0

KEYS = ["date", "city", "ward_id"]

MODEL_COLUMNS = {
    "A_original": ("A", "predicted_cases"),
    "B_unweighted": ("B", "prediction_weight_1_0"),
    "B_weight_1_5": ("B", "prediction_weight_1_5"),
    "B_weight_2_0": ("B", "prediction_weight_2_0"),
    "B_weight_3_0": ("B", "prediction_weight_3_0"),
    "C1_unweighted": ("C", "prediction_C1_unweighted"),
    "C2_weighted_2x": ("C", "prediction_C2_weighted_2x"),
}


# ============================================================
# LOAD AND ALIGN DATA
# ============================================================

def load_predictions():

    for path in [A_PATH, B_PATH, C_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    a = pd.read_csv(
        A_PATH,
        parse_dates=["date", "target_start_date", "target_end_date"],
    )

    b = pd.read_csv(
        B_PATH,
        parse_dates=["date", "target_start_date", "target_end_date"],
    )

    c = pd.read_csv(
        C_PATH,
        parse_dates=["date", "target_start_date", "target_end_date"],
    )

    a = a.sort_values(KEYS).reset_index(drop=True)
    b = b.sort_values(KEYS).reset_index(drop=True)
    c = c.sort_values(KEYS).reset_index(drop=True)

    # Check uniqueness before merging.
    for name, frame in [("A", a), ("B", b), ("C", c)]:
        if frame.duplicated(KEYS).any():
            raise ValueError(
                f"Duplicate prediction rows found in Experiment {name}."
            )

    # Experiment A is the reference row set.
    required_a = KEYS + [
        "target_start_date",
        "target_end_date",
        "actual_cases",
        "predicted_cases",
        "baseline_predictions",
    ]

    required_b = KEYS + [
        "target_start_date",
        "target_end_date",
        "actual_cases",
        "prediction_weight_1_0",
        "prediction_weight_1_5",
        "prediction_weight_2_0",
        "prediction_weight_3_0",
    ]

    required_c = KEYS + [
        "target_start_date",
        "target_end_date",
        "actual_cases",
        "prediction_C1_unweighted",
        "prediction_C2_weighted_2x",
    ]

    for name, frame, required in [
        ("A", a, required_a),
        ("B", b, required_b),
        ("C", c, required_c),
    ]:
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(
                f"Experiment {name} is missing columns: {missing}"
            )

    # Compare the row identities.
    if not a[KEYS].equals(b[KEYS]):
        raise ValueError("Experiment A and B validation rows do not match.")

    if not a[KEYS].equals(c[KEYS]):
        raise ValueError("Experiment A and C validation rows do not match.")

    # Compare target dates and actual outcomes.
    for other_name, other in [("B", b), ("C", c)]:

        if not a["target_start_date"].equals(
            other["target_start_date"]
        ):
            raise ValueError(
                f"Target start dates do not match for Experiment {other_name}."
            )

        if not a["target_end_date"].equals(
            other["target_end_date"]
        ):
            raise ValueError(
                f"Target end dates do not match for Experiment {other_name}."
            )

        if not np.allclose(
            a["actual_cases"].to_numpy(dtype=float),
            other["actual_cases"].to_numpy(dtype=float),
        ):
            raise ValueError(
                f"Actual outcomes do not match for Experiment {other_name}."
            )

    # Combine all predictions.
    combined = a[required_a].copy()

    for col in required_b:
        if col not in KEYS + [
            "target_start_date",
            "target_end_date",
            "actual_cases",
        ]:
            combined[col] = b[col].to_numpy()

    for col in required_c:
        if col not in KEYS + [
            "target_start_date",
            "target_end_date",
            "actual_cases",
        ]:
            combined[col] = c[col].to_numpy()

    return combined


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float),
        0,
    )

    actual_high = actual >= HIGH_THRESHOLD
    predicted_high = predicted >= HIGH_THRESHOLD

    tp = int(np.sum(actual_high & predicted_high))
    fp = int(np.sum(~actual_high & predicted_high))
    fn = int(np.sum(actual_high & ~predicted_high))

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else None
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else None
    )

    if precision is not None and recall is not None:
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
    else:
        f1 = None

    return {
        "records": int(len(actual)),
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(
            float(np.sqrt(mean_squared_error(actual, predicted))),
            4,
        ),
        "bias": round(float(np.mean(predicted - actual)), 4),
        "actual_high_records": int(np.sum(actual_high)),
        "predicted_high_records": int(np.sum(predicted_high)),
        "true_positive": tp,
        "false_positive": fp,
        "missed_high_events": fn,
        "precision": (
            round(float(precision), 4)
            if precision is not None else None
        ),
        "recall": (
            round(float(recall), 4)
            if recall is not None else None
        ),
        "f1": (
            round(float(f1), 4)
            if f1 is not None else None
        ),
        "miss_rate": (
            round(float(1 - recall), 4)
            if recall is not None else None
        ),
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 90)
    print("DENGUEGUARD AI - TARGET MONTH DIAGNOSTICS")
    print("=" * 90)

    df = load_predictions()

    # The target consists of cases at t+2, t+3 and t+4.
    # Use the middle target week as the seasonal grouping month.
    midpoint = (
        df["target_start_date"]
        + (df["target_end_date"] - df["target_start_date"]) / 2
    )

    df["target_month"] = midpoint.dt.month
    df["target_start_month"] = df["target_start_date"].dt.month
    df["target_end_month"] = df["target_end_date"].dt.month

    model_predictions = {
        name: df[column].to_numpy(dtype=float)
        for name, (_, column) in MODEL_COLUMNS.items()
    }

    model_predictions["Baseline"] = (
        df["baseline_predictions"].to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # DATA VALIDATION
    # --------------------------------------------------------

    print("\nVALIDATION DATA")
    print("-" * 90)

    print("Records:", len(df))
    print("High-incidence threshold:", HIGH_THRESHOLD)
    print("Target window:", "t+2, t+3 and t+4")
    print("Grouping:", "Month of middle target week")

    print(
        "Target start:",
        df["target_start_date"].min().date(),
        "to",
        df["target_start_date"].max().date(),
    )

    print(
        "Target end:",
        df["target_end_date"].min().date(),
        "to",
        df["target_end_date"].max().date(),
    )

    print(
        "Actual validation rows with high incidence:",
        int((df["actual_cases"] >= HIGH_THRESHOLD).sum()),
    )

    # --------------------------------------------------------
    # OVERALL RESULTS
    # --------------------------------------------------------

    overall_results = {}

    for model_name, predictions in model_predictions.items():

        overall_results[model_name] = calculate_metrics(
            df["actual_cases"],
            predictions,
        )

    # --------------------------------------------------------
    # TARGET MONTH RESULTS
    # --------------------------------------------------------

    monthly_results = []
    json_monthly_results = {}

    for month in range(1, 13):

        month_df = df[df["target_month"] == month]

        if month_df.empty:
            continue

        month_name = pd.Timestamp(
            year=2024,
            month=month,
            day=1,
        ).strftime("%B")

        json_monthly_results[month_name] = {}

        for model_name, predictions in model_predictions.items():

            indices = month_df.index.to_numpy()

            actual = month_df["actual_cases"].to_numpy(dtype=float)
            predicted = predictions[indices]

            metrics = calculate_metrics(
                actual,
                predicted,
            )

            json_monthly_results[month_name][model_name] = metrics

            monthly_results.append({
                "target_month": month_name,
                "model": model_name,
                **metrics,
            })

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    monthly_df = pd.DataFrame(monthly_results)

    monthly_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    report = {
        "analysis": "Target-month comparison of Experiments A, B and C",
        "validation_records": int(len(df)),
        "high_incidence_threshold": HIGH_THRESHOLD,
        "target_definition": "Sum of cases at t+2, t+3 and t+4",
        "month_definition": (
            "Calendar month of the midpoint between target_start_date "
            "and target_end_date"
        ),
        "test_set_used": False,
        "overall_results": overall_results,
        "monthly_results": json_monthly_results,
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

    # --------------------------------------------------------
    # PRINT OVERALL COMPARISON
    # --------------------------------------------------------

    print("\nOVERALL MODEL COMPARISON")
    print("-" * 110)

    print(
        f"{'Model':<22}"
        f"{'MAE':<12}"
        f"{'RMSE':<12}"
        f"{'Bias':<12}"
        f"{'Precision':<13}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Miss Rate':<12}"
    )

    for model_name, metrics in overall_results.items():

        def fmt(value):
            return f"{value:.4f}" if value is not None else "N/A"

        print(
            f"{model_name:<22}"
            f"{metrics['mae']:<12.4f}"
            f"{metrics['rmse']:<12.4f}"
            f"{metrics['bias']:<12.4f}"
            f"{fmt(metrics['precision']):<13}"
            f"{fmt(metrics['recall']):<12}"
            f"{fmt(metrics['f1']):<12}"
            f"{fmt(metrics['miss_rate']):<12}"
        )

    # --------------------------------------------------------
    # PRINT MONTHLY MAE
    # --------------------------------------------------------

    print("\nMONTHLY MAE - GROUPED BY TARGET MONTH")
    print("-" * 110)

    pivot_mae = monthly_df.pivot(
        index="target_month",
        columns="model",
        values="mae",
    )

    print(pivot_mae.round(4).to_string())

    # --------------------------------------------------------
    # PRINT MONTHLY HIGH-INCIDENCE RECALL
    # --------------------------------------------------------

    print("\nMONTHLY HIGH-INCIDENCE RECALL")
    print("-" * 110)

    pivot_recall = monthly_df.pivot(
        index="target_month",
        columns="model",
        values="recall",
    )

    print(pivot_recall.round(4).to_string())

    # --------------------------------------------------------
    # PRINT MONTHLY BIAS
    # --------------------------------------------------------

    print("\nMONTHLY BIAS")
    print("-" * 110)

    pivot_bias = monthly_df.pivot(
        index="target_month",
        columns="model",
        values="bias",
    )

    print(pivot_bias.round(4).to_string())

    # --------------------------------------------------------
    # SAVED FILES
    # --------------------------------------------------------

    print("\nSAVED FILES")
    print("-" * 90)

    print("CSV:", OUTPUT_CSV)
    print("JSON:", OUTPUT_JSON)

    print("\nTARGET MONTH ANALYSIS COMPLETED.")
    print("No models or test predictions were modified.")


if __name__ == "__main__":
    main()
