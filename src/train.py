
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dengue_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "dengue_model.joblib"
METRICS_PATH = REPORT_DIR / "model_evaluation.json"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"

TARGET = "target_cases_next_2_4w"

VALIDATION_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")

CATEGORICAL_FEATURES = ["city", "ward_id"]

EXCLUDED_COLUMNS = [
    "date",
    "dengue_cases",
    TARGET,
    "target_start_date",
    "target_end_date"
]


def calculate_metrics(actual, predicted):

    predicted = np.maximum(predicted, 0)

    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
        "r2": round(float(r2_score(actual, predicted)), 4)
    }


def create_pipeline(model, numeric_features):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                "passthrough",
                numeric_features
            )
        ],
        remainder="drop"
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ]
    )


def main():

    print("=" * 65)
    print("DENGUEGUARD AI - MODEL TRAINING")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date", "target_start_date", "target_end_date"]
    )

    df = df.sort_values(
        ["date", "ward_id"]
    ).reset_index(drop=True)

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    numeric_features = [
        column
        for column in feature_columns
        if column not in CATEGORICAL_FEATURES
    ]

    if not all(column in df.columns for column in CATEGORICAL_FEATURES):
        raise ValueError("Categorical feature columns are missing.")

    if df[feature_columns].isnull().any().any():
        raise ValueError("Feature dataset contains missing input values.")

    if df[TARGET].isnull().any():
        raise ValueError("Target column contains missing values.")

    train_mask = (
        (df["date"] < VALIDATION_START)
        & (df["target_end_date"] < VALIDATION_START)
    )

    validation_mask = (
        (df["date"] >= VALIDATION_START)
        & (df["date"] < TEST_START)
        & (df["target_end_date"] < TEST_START)
    )

    test_mask = df["date"] >= TEST_START

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()
    test_df = df.loc[test_mask].copy()

    if min(len(train_df), len(validation_df), len(test_df)) == 0:
        raise ValueError("One or more chronological datasets are empty.")

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df[TARGET]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    print("\nCHRONOLOGICAL DATA SPLIT")
    print("-" * 65)
    print(f"Training records: {len(train_df)}")
    print(f"Validation records: {len(validation_df)}")
    print(f"Testing records: {len(test_df)}")
    print(f"Input features: {len(feature_columns)}")

    print("\nTraining date range:")
    print(train_df["date"].min(), "to", train_df["date"].max())

    print("\nValidation date range:")
    print(validation_df["date"].min(), "to", validation_df["date"].max())

    print("\nTesting date range:")
    print(test_df["date"].min(), "to", test_df["date"].max())

    baseline_validation = (
        validation_df["dengue_cases_rolling_4w"].to_numpy() * 3
    )

    baseline_test = (
        test_df["dengue_cases_rolling_4w"].to_numpy() * 3
    )

    validation_results = {
        "Baseline": calculate_metrics(
            y_validation,
            baseline_validation
        )
    }

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        )
    }

    trained_validation_models = {}

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        pipeline = create_pipeline(
            model,
            numeric_features
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_validation)

        validation_results[model_name] = calculate_metrics(
            y_validation,
            predictions
        )

        trained_validation_models[model_name] = pipeline

        print(
            f"{model_name} validation MAE:",
            validation_results[model_name]["mae"]
        )

        print(
            f"{model_name} validation RMSE:",
            validation_results[model_name]["rmse"]
        )

        print(
            f"{model_name} validation R2:",
            validation_results[model_name]["r2"]
        )

    candidate_names = list(models.keys())

    best_model_name = min(
        candidate_names,
        key=lambda name: validation_results[name]["mae"]
    )

    print("\nMODEL SELECTION")
    print("-" * 65)
    print("Baseline validation MAE:", validation_results["Baseline"]["mae"])
    print("Selected model:", best_model_name)

    final_training_df = df.loc[
        (df["date"] < TEST_START)
        & (df["target_end_date"] < TEST_START)
    ].copy()

    X_final_train = final_training_df[feature_columns]
    y_final_train = final_training_df[TARGET]

    final_model = create_pipeline(
        models[best_model_name],
        numeric_features
    )

    print("\nRetraining selected model on pre-test data...")

    final_model.fit(
        X_final_train,
        y_final_train
    )

    test_predictions = final_model.predict(X_test)

    test_predictions = np.maximum(
        test_predictions,
        0
    )

    test_results = calculate_metrics(
        y_test,
        test_predictions
    )

    baseline_test_results = calculate_metrics(
        y_test,
        baseline_test
    )

    predictions_df = pd.DataFrame({
        "date": test_df["date"].values,
        "city": test_df["city"].values,
        "ward_id": test_df["ward_id"].values,
        "actual_cases_next_2_4w": y_test.values,
        "predicted_cases_next_2_4w": np.round(
            test_predictions, 2
        ),
        "baseline_predictions": np.round(
            baseline_test, 2
        )
    })

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        final_model,
        MODEL_PATH
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    report = {
        "selected_model": best_model_name,
        "target": TARGET,
        "target_definition": "Sum of dengue cases in weeks t+2, t+3 and t+4",
        "training_records": int(len(train_df)),
        "validation_records": int(len(validation_df)),
        "test_records": int(len(test_df)),
        "final_training_records": int(len(final_training_df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "validation_metrics": validation_results,
        "test_metrics": {
            "selected_model": test_results,
            "baseline": baseline_test_results
        },
        "model_path": str(MODEL_PATH),
        "predictions_path": str(PREDICTIONS_PATH)
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\nFINAL TEST RESULTS")
    print("-" * 65)

    print("Selected model:", best_model_name)
    print("Model MAE:", test_results["mae"])
    print("Model RMSE:", test_results["rmse"])
    print("Model R2:", test_results["r2"])

    print("\nBaseline MAE:", baseline_test_results["mae"])
    print("Baseline RMSE:", baseline_test_results["rmse"])
    print("Baseline R2:", baseline_test_results["r2"])

    improvement = (
        (
            baseline_test_results["mae"] - test_results["mae"]
        ) / baseline_test_results["mae"] * 100
        if baseline_test_results["mae"] > 0
        else None
    )

    print(
        "MAE improvement over baseline:",
        f"{improvement:.2f}%" if improvement is not None else "N/A"
    )

    print("\nSAVED FILES")
    print("-" * 65)
    print("Model:", MODEL_PATH)
    print("Metrics:", METRICS_PATH)
    print("Predictions:", PREDICTIONS_PATH)

    print("\nMODEL TRAINING COMPLETED.")


if __name__ == "__main__":
    main()
