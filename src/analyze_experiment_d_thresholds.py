
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "reports" / "experiment_d_2023_calibration.csv"

ACTUAL_THRESHOLD = 28.0
CANDIDATES = np.arange(8.0, 40.01, 0.5)
PRECISION_FLOORS = [0.80, 0.85, 0.90]

df = pd.read_csv(INPUT)

actual = df["actual_cases"].to_numpy()
predicted = np.maximum(df["predicted_cases"].to_numpy(), 0)

actual_high = actual >= ACTUAL_THRESHOLD

results = []

for threshold in CANDIDATES:
    alerts = predicted >= threshold

    results.append({
        "threshold": threshold,
        "precision": precision_score(
            actual_high, alerts, zero_division=0
        ),
        "recall": recall_score(
            actual_high, alerts, zero_division=0
        ),
        "f1": f1_score(
            actual_high, alerts, zero_division=0
        ),
        "alerts": int(alerts.sum()),
        "true_positive": int((actual_high & alerts).sum()),
        "false_positive": int((~actual_high & alerts).sum()),
        "missed_events": int((actual_high & ~alerts).sum()),
    })

results = pd.DataFrame(results)

print("=" * 85)
print("EXPERIMENT D - PRECISION-CONSTRAINED THRESHOLD ANALYSIS")
print("=" * 85)

for floor in PRECISION_FLOORS:

    eligible = results[
        results["precision"] >= floor
    ].copy()

    print(f"\nMINIMUM PRECISION: {floor:.0%}")
    print("-" * 85)

    if eligible.empty:
        print("No threshold satisfies this precision requirement.")
        continue

    best = eligible.sort_values(
        ["recall", "f1", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]

    print(f"Selected threshold: {best['threshold']:.1f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"F1: {best['f1']:.4f}")
    print(f"Alerts: {int(best['alerts'])}")
    print(f"True positives: {int(best['true_positive'])}")
    print(f"False positives: {int(best['false_positive'])}")
    print(f"Missed events: {int(best['missed_events'])}")

print("\nEXISTING THRESHOLD: 28")
print("-" * 85)

existing = results[
    np.isclose(results["threshold"], 28.0)
].iloc[0]

print(existing.to_string())

print("\nTOP 10 THRESHOLDS BY F1")
print("-" * 85)

print(
    results.sort_values(
        ["f1", "precision"],
        ascending=False,
    ).head(10).to_string(index=False)
)
