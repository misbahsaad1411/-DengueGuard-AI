# DengueGuard AI
### AI-Powered Ward-Level Dengue Forecasting and Outbreak Risk Intelligence

DengueGuard AI is an AI-powered dengue forecasting and risk-intelligence prototype designed for Bangladesh. It combines historical dengue cases, weather patterns, and ward-level environmental indicators to forecast future dengue activity and identify areas that may require public-health attention.

> **Project status:** AI Engineering Hackathon research prototype. Predictions and risk levels are experimental decision-support signals, not official outbreak declarations or medical advice.

---

## 1. Problem Statement

Dengue outbreaks can vary significantly across geographical locations and seasons. Early identification of potentially high-risk areas can help public-health authorities prioritize surveillance, prevention, and resource allocation.

DengueGuard AI explores how machine learning, historical dengue records, weather patterns, and environmental indicators can be combined to provide ward-level dengue forecasts and risk intelligence.

## 2. Key Features

- **Ward-level forecasting:** Predicts dengue cases over a future multi-week period.
- **Weather intelligence:** Uses historical rainfall, temperature, and humidity patterns.
- **Environmental indicators:** Incorporates water-logging risk, sanitation, and population density.
- **Four-level risk classification:** Low, Guarded, High, and Critical.
- **Outbreak warning:** Includes a separate high-event warning classifier.
- **Interactive dashboard:** Explore predictions, risk distributions, cities, wards, and model performance.
- **Experimental evaluation:** Includes multiple model experiments and historical backtesting.

## 3. Dataset Overview

The project uses the provided Bangladesh dengue hackathon dataset.

| Attribute | Description |
|---|---|
| Total records | 31,320 |
| Cities | 5 |
| Wards | 60 |
| Time period | 2016–2025 |
| Frequency | Weekly |
| Engineered records | 30,840 |
| Model features | 46 |

### Cities covered

- Dhaka North
- Dhaka South
- Chattogram
- Gazipur
- Narayanganj

### Dataset features

- Dengue case counts
- Population density
- Water-logging risk index
- Sanitation score
- Average, maximum, and minimum temperature
- Rainfall
- Relative humidity

## 4. Feature Engineering

The feature engineering pipeline creates 46 model features.

### Weather lag features

Historical weather information from the previous four weeks:

- Rainfall
- Average temperature
- Maximum temperature
- Minimum temperature
- Relative humidity

### Dengue history features

- Dengue cases at lags 1–4 weeks
- Previous four-week rolling mean
- Previous four-week rolling sum

### Additional features

- Calendar week and month
- Cyclical calendar encodings
- City and ward identifiers

Rolling features use historical observations to avoid using future information.

### Forecasting target

The model predicts the cumulative dengue cases over weeks t+2, t+3, and t+4:

`Target = Cases(t+2) + Cases(t+3) + Cases(t+4)`

Week t+1 is intentionally excluded.

## 5. Machine Learning Architecture

### Selected forecasting model

**HistGradientBoostingRegressor with Quantile Loss (Q50)**

The selected model was chosen after comparing multiple regression approaches, quantile models, severity weighting, model blending, and rolling-origin validation.

Model artifacts:

- `models/dengueguard_q50_forecaster.joblib`
- `models/dengueguard_metadata.json`

### Separate outbreak-warning model

A Random Forest classifier provides an additional high-event warning probability.

Artifact:

- `models/dengueguard_high_warning.joblib`

The warning classifier is supplementary to the forecasting model and serves a different purpose.

## 6. Model Evaluation

The project uses chronological validation and historical rolling-origin backtesting.

### 2025 regression evaluation

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Original Model A | 2.6946 | 4.9629 | 0.7880 |
| **Selected Q50** | **2.6069** | **4.7471** | **0.8060** |
| 50/50 Blend | 2.6160 | 4.7705 | 0.8041 |

The selected Q50 model achieved approximately **3.25% lower MAE** than the original Model A on the 2025 evaluation set.

### Historical rolling-origin evaluation (2019–2023)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Original Model A | 3.0250 | 5.7090 | 0.8234 |
| **Q50** | **2.8103** | **5.3462** | **0.8451** |

Q50 achieved approximately 7.10% lower pooled MAE than Model A across these historical validation folds.

*Note: R² is a regression goodness-of-fit metric, not classification accuracy.*

### High-event warning evaluation (2025)

At a probability threshold of 0.50:

| Metric | Result |
|---|---:|
| Precision | 0.442 |
| Recall | 0.876 |
| F1 Score | 0.588 |
| True Positives | 211 |
| False Positives | 266 |
| False Negatives | 30 |

The classifier prioritizes recall, but its false-positive count is substantial. It should be treated as an additional screening signal.

## 7. Risk Classification

DengueGuard AI uses four risk categories:

| Risk Level | Interpretation |
|---|---|
| Low | Relatively low predicted activity |
| Guarded | Elevated compared with the low-risk baseline |
| High | Higher historical relative risk |
| Critical | Highest relative risk category |

Risk thresholds are derived from historical positive-case distributions using city-month reference values where available, with fallback thresholds when needed.

**Important:** These are experimental project-defined categories, not official Bangladesh government outbreak-alert levels. They are not population-adjusted incidence rates.

## 8. Interactive Dashboard

The Streamlit dashboard provides:

- Interactive city and ward filters
- Risk-level selection
- Prediction date filtering
- Risk distribution visualizations
- City-wise and ward-wise analysis
- Predicted versus actual comparisons
- Model performance metrics
- Weather and environmental insights
- Prediction tables and CSV export

The current dashboard provides retrospective exploration of saved historical predictions. It is not a live surveillance system.

## 9. Project Structure

```text
DengueGuard-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── dengue_model.joblib
│   ├── dengueguard_q50_forecaster.joblib
│   ├── dengueguard_high_warning.joblib
│   └── dengueguard_metadata.json
│
├── reports/
│   ├── experiment_p_2025_predictions.csv
│   ├── experiment_p_2025_regression.csv
│   ├── experiment_p_2025_high_event.csv
│   └── ... additional experiment reports
│
└── src/
    ├── feature_engineering.py
    ├── train.py
    ├── risk_engine.py
    ├── experiment_j_quantile.py
    ├── experiment_k_robustness.py
    ├── experiment_l_rolling_backtest.py
    ├── experiment_m_model_blending.py
    ├── experiment_n_severe_diagnostics.py
    ├── experiment_o_outbreak_classifier.py
    └── experiment_p_final_selection.py
```

## 10. Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- pip

### Step 1: Clone the repository

```bash
git clone https://github.com/misbahsaad1411/-DengueGuard-AI.git
cd -DengueGuard-AI
```

### Step 2: Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the dashboard

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit, usually:

`http://localhost:8501`

## 11. Reproducibility

The repository includes:

- Feature engineering scripts
- Model training scripts
- Historical validation scripts
- Experimental model comparisons
- Saved model artifacts
- Prediction outputs
- Evaluation reports

The final model selection was based on chronological evaluation rather than random splitting.

Re-running all experiments may require additional memory and processing time.

## 12. Limitations

DengueGuard AI is an experimental prototype with important limitations.

1. **Severe outbreak prediction:** Severe-event performance remains weak. Only 11 severe records (target cases ≥45) occurred in the 2025 evaluation set.
2. **False alarms:** The recall-oriented warning classifier produces substantial false positives.
3. **Historical dependency:** Changes in climate, public-health interventions, reporting, and population patterns may affect model performance.
4. **Relative risk thresholds:** The four risk categories are based on historical distributions and are not official public-health alert levels.
5. **Retrospective dashboard:** The current dashboard visualizes saved historical predictions rather than live surveillance data.
6. **Geographical coverage:** Results should not be generalized to locations outside the dataset without further validation.

## 13. Responsible Use

DengueGuard AI is intended to support research, analysis, and public-health planning.

It should not independently determine public-health interventions, diagnose patients, or replace expert judgment.

Operational deployment would require additional external validation, ongoing monitoring, reliable real-time data, and collaboration with public-health professionals.

## 14. Hackathon Project

**Project:** DengueGuard AI  
**Category:** AI Engineering / Machine Learning / Public Health  
**Focus:** Ward-level dengue forecasting and outbreak risk intelligence  
**Technology:** Python, Scikit-learn, Pandas, NumPy, Streamlit, Plotly

---

*Built as an AI Engineering Hackathon project.*

*Dataset redistribution and external AI/API usage should comply with the hackathon rules and applicable data licenses.*
