
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"

DATA_PATH = (
    ROOT / "data" / "processed" / "dengue_features.csv"
)

L_PREDICTIONS = (
    REPORT_DIR / "experiment_l_predictions.csv"
)

OUTPUT_PREDICTIONS = (
    REPORT_DIR / "experiment_o_predictions.csv"
)

OUTPUT_SUMMARY = (
    REPORT_DIR / "experiment_o_summary.json"
)

OUTPUT_METRICS = (
    REPORT_DIR / "experiment_o_metrics.csv"
)

VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]

SEVERE_THRESHOLD = 45.0

PROBABILITY_CUTOFFS = [0.25, 0.35, 0.50]

RANDOM_STATE = 42

TARGET = "target_cases_next_2_4w"

FEATURE_EXCLUSIONS = [
    "date",
    "dengue_cases",
    TARGET,
    "target_start_date",
    "target_end_date",
]


# ============================================================
# CLASSIFIER
# ============================================================

def create_classifier():

    categorical = ["city", "ward_id"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            )
        ],
        remainder="passthrough",
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=4,
        max_features=0.8,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ============================================================
# METRICS
# ============================================================

def classification_metrics(
    actual,
    predicted,
    scores,
):

    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    scores = np.asarray(scores, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[False, True],
    ).ravel()

    precision = precision_score(
        actual,
        predicted,
        zero_division=0,
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0,
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0,
    )

    if actual.sum() > 0:
        average_precision = average_precision_score(
            actual,
            scores,
        )
    else:
        average_precision = None

    return {
        "records": int(len(actual)),
        "actual_positive": int(actual.sum()),
        "predicted_positive": int(predicted.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": (
            float(average_precision)
            if average_precision is not None
            else None
        ),
    }


def add_result(
    results,
    year,
    model_name,
    event_type,
    cutoff,
    actual,
    scores,
):

    predicted = scores >= cutoff

    metrics = classification_metrics(
        actual,
        predicted,
        scores,
    )

    results.append({
        "year": year,
        "model": model_name,
        "event_type": event_type,
        "probability_cutoff": cutoff,
        **metrics,
    })

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 95)
    print("DENGUEGUARD AI - EXPERIMENT O")
    print("SEPARATE OUTBREAK-WARNING CLASSIFIERS")
    print("=" * 95)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "date",
            "target_start_date",
            "target_end_date",
        ],
    )

    df = df.sort_values(
        ["date", "ward_id"]
    ).reset_index(drop=True)

    if df.duplicated(
        subset=["date", "city", "ward_id"]
    ).any():

        raise ValueError(
            "Duplicate date/city/ward combinations found "
            "in the feature dataset."
        )

    X = df[
        [
            column
            for column in df.columns
            if column not in FEATURE_EXCLUSIONS
        ]
    ].copy()

    y = df[TARGET].astype(float)

    # --------------------------------------------------------
    # LOAD EXPERIMENT L PREDICTIONS
    # --------------------------------------------------------

    predictions = pd.read_csv(
        L_PREDICTIONS,
        parse_dates=["date"],
    )

    prediction_columns = [
        "date",
        "city",
        "ward_id",
        "model_A",
        "Q50",
    ]

    missing_columns = [
        column
        for column in prediction_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in Experiment L predictions: "
            f"{missing_columns}"
        )

    predictions = predictions[
        prediction_columns
    ].copy()

    if predictions.duplicated(
        subset=["date", "city", "ward_id"]
    ).any():

        raise ValueError(
            "Duplicate prediction keys in Experiment L file."
        )

    # --------------------------------------------------------
    # MERGE BASELINE PREDICTIONS
    # --------------------------------------------------------

    df = df.merge(
        predictions,
        on=["date", "city", "ward_id"],
        how="left",
        validate="one_to_one",
    )

    print(f"\nDataset records: {len(df):,}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Validation years: {VALIDATION_YEARS}")

    all_fold_predictions = []
    metric_results = []

    # ========================================================
    # ROLLING-ORIGIN EVALUATION
    # ========================================================

    for year in VALIDATION_YEARS:

        print("\n" + "=" * 95)
        print(f"VALIDATION YEAR: {year}")
        print("=" * 95)

        year_start = pd.Timestamp(
            f"{year}-01-01"
        )

        next_year_start = pd.Timestamp(
            f"{year + 1}-01-01"
        )

        # Same split as Experiment L.
        train_mask = (
            (df["date"] < year_start)
            & (df["target_end_date"] < year_start)
        )

        validation_mask = (
            (df["date"] >= year_start)
            & (df["date"] < next_year_start)
            & (df["target_end_date"] < next_year_start)
        )

        X_train = X.loc[train_mask]
        X_val = X.loc[validation_mask]

        y_train = y.loc[
            train_mask
        ].to_numpy(dtype=float)

        y_val = y.loc[
            validation_mask
        ].to_numpy(dtype=float)

        validation_df = df.loc[
            validation_mask,
            [
                "date",
                "city",
                "ward_id",
                "model_A",
                "Q50",
            ],
        ].copy()

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(
                f"Empty training or validation data for {year}"
            )

        # IMPORTANT:
        # Check baseline predictions only in the exact
        # validation fold, not across the entire dataset.

        if validation_df[
            ["model_A", "Q50"]
        ].isna().any().any():

            raise ValueError(
                f"Missing baseline predictions inside "
                f"Experiment L validation fold {year}."
            )

        # ----------------------------------------------------
        # TRAINING-ONLY HIGH EVENT THRESHOLD
        # ----------------------------------------------------

        high_threshold = float(
            np.quantile(y_train, 0.90)
        )

        print(f"Training records: {len(X_train):,}")
        print(f"Validation records: {len(X_val):,}")
        print(
            f"Training 90th percentile: "
            f"{high_threshold:.2f}"
        )

        # ----------------------------------------------------
        # CREATE TARGET LABELS
        # ----------------------------------------------------

        y_train_high = (
            y_train >= high_threshold
        ).astype(int)

        y_val_high = (
            y_val >= high_threshold
        )

        y_train_severe = (
            y_train >= SEVERE_THRESHOLD
        ).astype(int)

        y_val_severe = (
            y_val >= SEVERE_THRESHOLD
        )

        print(
            f"Training high events: "
            f"{y_train_high.sum():,}"
        )

        print(
            f"Validation high events: "
            f"{y_val_high.sum():,}"
        )

        print(
            f"Training severe events: "
            f"{y_train_severe.sum():,}"
        )

        print(
            f"Validation severe events: "
            f"{y_val_severe.sum():,}"
        )

        validation_df["actual"] = y_val
        validation_df["high_threshold"] = high_threshold

        validation_df["actual_high"] = y_val_high
        validation_df["actual_severe"] = y_val_severe

        # ====================================================
        # MODEL A BASELINE
        # ====================================================

        model_a_scores = (
            validation_df["model_A"].to_numpy(dtype=float)
        )

        model_a_high = (
            model_a_scores >= high_threshold
        )

        model_a_high_metrics = classification_metrics(
            y_val_high,
            model_a_high,
            model_a_scores,
        )

        metric_results.append({
            "year": year,
            "model": "model_A",
            "event_type": "high_baseline",
            "probability_cutoff": high_threshold,
            **model_a_high_metrics,
        })

        model_a_severe = (
            model_a_scores >= SEVERE_THRESHOLD
        )

        model_a_severe_metrics = classification_metrics(
            y_val_severe,
            model_a_severe,
            model_a_scores,
        )

        metric_results.append({
            "year": year,
            "model": "model_A",
            "event_type": "severe_baseline",
            "probability_cutoff": SEVERE_THRESHOLD,
            **model_a_severe_metrics,
        })

        # ====================================================
        # Q50 BASELINE
        # ====================================================

        q50_scores = (
            validation_df["Q50"].to_numpy(dtype=float)
        )

        q50_high_metrics = classification_metrics(
            y_val_high,
            q50_scores >= high_threshold,
            q50_scores,
        )

        metric_results.append({
            "year": year,
            "model": "Q50",
            "event_type": "high_baseline",
            "probability_cutoff": high_threshold,
            **q50_high_metrics,
        })

        q50_severe_metrics = classification_metrics(
            y_val_severe,
            q50_scores >= SEVERE_THRESHOLD,
            q50_scores,
        )

        metric_results.append({
            "year": year,
            "model": "Q50",
            "event_type": "severe_baseline",
            "probability_cutoff": SEVERE_THRESHOLD,
            **q50_severe_metrics,
        })

        # ====================================================
        # HIGH-EVENT CLASSIFIER
        # ====================================================

        print("\nTraining high-event classifier...")

        if len(np.unique(y_train_high)) < 2:
            raise ValueError(
                f"High-event training labels contain only "
                f"one class in {year}."
            )

        high_model = create_classifier()

        high_model.fit(
            X_train,
            y_train_high,
        )

        high_probability = high_model.predict_proba(
            X_val
        )[:, 1]

        validation_df["high_event_probability"] = (
            high_probability
        )

        for cutoff in PROBABILITY_CUTOFFS:

            metrics = add_result(
                metric_results,
                year,
                "high_classifier",
                "high_event",
                cutoff,
                y_val_high,
                high_probability,
            )

            print(
                f"High classifier | cutoff={cutoff:.2f} | "
                f"precision={metrics['precision']:.3f} | "
                f"recall={metrics['recall']:.3f} | "
                f"F1={metrics['f1']:.3f}"
            )

        del high_model
        gc.collect()

        # ====================================================
        # SEVERE-EVENT CLASSIFIER
        # ====================================================

        print("\nTraining severe-event classifier...")

        if len(np.unique(y_train_severe)) < 2:

            print(
                "WARNING: Severe training labels contain "
                "only one class. Skipping this fold."
            )

            validation_df[
                "severe_event_probability"
            ] = np.nan

        else:

            severe_model = create_classifier()

            severe_model.fit(
                X_train,
                y_train_severe,
            )

            severe_probability = (
                severe_model.predict_proba(X_val)[:, 1]
            )

            validation_df[
                "severe_event_probability"
            ] = severe_probability

            for cutoff in PROBABILITY_CUTOFFS:

                metrics = add_result(
                    metric_results,
                    year,
                    "severe_classifier",
                    "severe_event",
                    cutoff,
                    y_val_severe,
                    severe_probability,
                )

                print(
                    f"Severe classifier | cutoff={cutoff:.2f} | "
                    f"precision={metrics['precision']:.3f} | "
                    f"recall={metrics['recall']:.3f} | "
                    f"F1={metrics['f1']:.3f}"
                )

            del severe_model
            gc.collect()

        all_fold_predictions.append(
            validation_df
        )

        del X_train, X_val
        gc.collect()

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_output = pd.concat(
        all_fold_predictions,
        ignore_index=True,
    )

    prediction_output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_df = pd.DataFrame(
        metric_results
    )

    metrics_df.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    # ========================================================
    # POOLED METRICS
    # ========================================================

    aggregate = []

    grouping_columns = [
        "model",
        "event_type",
        "probability_cutoff",
    ]

    for keys, group in metrics_df.groupby(
        grouping_columns,
        dropna=False,
    ):

        model_name, event_type, cutoff = keys

        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        tn = int(group["tn"].sum())

        precision = (
            tp / (tp + fp)
            if tp + fp > 0 else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0 else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )

        ap_values = (
            group["average_precision"]
            .dropna()
        )

        aggregate.append({
            "model": model_name,
            "event_type": event_type,
            "probability_cutoff": (
                float(cutoff)
                if pd.notna(cutoff)
                else None
            ),
            "records": tp + fp + fn + tn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_yearly_average_precision": (
                float(ap_values.mean())
                if len(ap_values) > 0
                else None
            ),
        })

    aggregate_df = pd.DataFrame(
        aggregate
    )

    print("\n" + "=" * 95)
    print("POOLED OUTBREAK CLASSIFICATION RESULTS")
    print("=" * 95)

    print(
        aggregate_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # ========================================================
    # YEARLY RESULTS
    # ========================================================

    print("\nYEARLY CLASSIFIER RESULTS")
    print("-" * 95)

    selected = metrics_df[
        metrics_df["model"].isin([
            "high_classifier",
            "severe_classifier",
        ])
    ]

    yearly_columns = [
        "year",
        "model",
        "event_type",
        "probability_cutoff",
        "actual_positive",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "average_precision",
    ]

    print(
        selected[yearly_columns].to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {
        "experiment": (
            "O - Separate Outbreak-Warning Classifiers"
        ),
        "validation_years": VALIDATION_YEARS,
        "severe_threshold": SEVERE_THRESHOLD,
        "probability_cutoffs": PROBABILITY_CUTOFFS,
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 250,
            "min_samples_leaf": 4,
            "max_features": 0.8,
            "class_weight": "balanced_subsample",
            "random_state": RANDOM_STATE,
        },
        "pooled_results": aggregate,
        "yearly_results": metric_results,
    }

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\nSAVED REPORTS")
    print(OUTPUT_PREDICTIONS)
    print(OUTPUT_METRICS)
    print(OUTPUT_SUMMARY)

    print("\nEXPERIMENT O COMPLETE")


if __name__ == "__main__":
    main()
