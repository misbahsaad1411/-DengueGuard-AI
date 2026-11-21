from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# DENGUEGUARD AI - FINAL RETROSPECTIVE DASHBOARD
# ============================================================

st.set_page_config(
    page_title="DengueGuard AI | Early Warning Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
PREDICTION_FILE = ROOT / "reports" / "experiment_p_2025_predictions.csv"
FEATURE_FILE = ROOT / "data" / "processed" / "dengue_features.csv"
METADATA_FILE = ROOT / "models" / "dengueguard_metadata.json"
FORECAST_MODEL_FILE = ROOT / "models" / "dengueguard_q50_forecaster.joblib"
WARNING_MODEL_FILE = ROOT / "models" / "dengueguard_high_warning.joblib"


# ============================================================
# PREMIUM UI THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --dg-bg: #07111f;
        --dg-panel: rgba(15, 29, 49, .82);
        --dg-panel-2: rgba(21, 40, 64, .72);
        --dg-border: rgba(148, 177, 211, .16);
        --dg-text: #edf5ff;
        --dg-muted: #9fb2c9;
        --dg-teal: #2dd4bf;
        --dg-blue: #60a5fa;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(13, 148, 136, .16), transparent 34%),
            radial-gradient(ellipse at 92% 4%, rgba(59, 130, 246, .14), transparent 32%),
            linear-gradient(145deg, #07111f 0%, #0a1728 48%, #0b1423 100%);
        color: var(--dg-text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, .72);
        backdrop-filter: blur(14px);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1a2c 0%, #0a1525 100%);
        border-right: 1px solid var(--dg-border);
    }

    [data-testid="stSidebar"] * {
        color: #dce9f8;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -.035em;
        color: #f4f8ff !important;
    }

    h1 { font-weight: 800 !important; }
    h2 { font-weight: 750 !important; margin-top: 1.1rem !important; }
    h3 { font-weight: 700 !important; }

    p, label, [data-testid="stMarkdownContainer"] {
        color: #d4e1f0;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(20, 43, 66, .92), rgba(13, 29, 48, .92));
        border: 1px solid rgba(125, 164, 201, .19);
        border-radius: 17px;
        padding: 17px 18px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, .16);
        transition: transform .18s ease, border-color .18s ease;
        min-height: 112px;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(45, 212, 191, .55);
    }

    [data-testid="stMetricLabel"] {
        color: #9fb5cc !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #f5fbff !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }

    [data-testid="stPlotlyChart"] {
        background: rgba(13, 28, 46, .72);
        border: 1px solid var(--dg-border);
        border-radius: 18px;
        padding: 8px;
        box-shadow: 0 12px 30px rgba(0,0,0,.12);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--dg-border);
        border-radius: 14px;
        overflow: hidden;
    }

    div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stAlert"]) {
        border-radius: 15px;
    }

    [data-testid="stAlert"] {
        border-radius: 15px;
        border: 1px solid rgba(96, 165, 250, .25);
    }

    .stButton > button, .stDownloadButton > button {
        border: 1px solid rgba(45, 212, 191, .38);
        background: linear-gradient(110deg, #0f766e, #0e7490);
        color: white;
        border-radius: 12px;
        font-weight: 700;
        padding: .55rem 1rem;
        transition: all .18s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #5eead4;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(13, 148, 136, .22);
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stDateInput"] input {
        background: rgba(16, 34, 55, .95);
        border-color: rgba(148, 177, 211, .24);
        border-radius: 10px;
    }

    [data-testid="stExpander"] {
        background: rgba(14, 29, 48, .72);
        border: 1px solid var(--dg-border);
        border-radius: 14px;
    }

    hr {
        border-color: rgba(148, 177, 211, .15);
        margin: 1.7rem 0;
    }

    .dg-hero {
        position: relative;
        overflow: hidden;
        padding: 28px 30px;
        margin: 0 0 22px 0;
        border: 1px solid rgba(94, 234, 212, .22);
        border-radius: 24px;
        background:
            radial-gradient(circle at 88% 10%, rgba(45, 212, 191, .20), transparent 27%),
            radial-gradient(circle at 72% 100%, rgba(59, 130, 246, .20), transparent 32%),
            linear-gradient(115deg, rgba(14, 45, 62, .98), rgba(15, 35, 60, .96));
        box-shadow: 0 18px 50px rgba(0,0,0,.22);
    }

    .dg-kicker {
        color: #5eead4;
        text-transform: uppercase;
        letter-spacing: .17em;
        font-size: .72rem;
        font-weight: 800;
        margin-bottom: 9px;
    }

    .dg-hero-title {
        color: #f4fbff;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3.1rem);
        line-height: 1.08;
        font-weight: 800;
        letter-spacing: -.055em;
        margin-bottom: 10px;
    }

    .dg-hero-sub {
        color: #c0d5e8;
        font-size: 1rem;
        max-width: 760px;
        line-height: 1.6;
    }

    .dg-chip {
        display: inline-block;
        margin-top: 17px;
        margin-right: 8px;
        padding: 6px 11px;
        border-radius: 999px;
        border: 1px solid rgba(94, 234, 212, .28);
        background: rgba(13, 148, 136, .13);
        color: #b8fff2;
        font-size: .76rem;
        font-weight: 700;
    }

    .dg-section-note {
        color: #91a9c2;
        font-size: .88rem;
        margin-top: -8px;
        margin-bottom: 14px;
    }

    .dg-footer {
        color: #8097b0;
        text-align: center;
        font-size: .82rem;
        padding: 22px 8px 4px 8px;
    }

    @media (max-width: 700px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .dg-hero { padding: 22px 20px; border-radius: 18px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#d8e7f5"),
    margin=dict(l=24, r=24, t=58, b=28),
    hoverlabel=dict(
        bgcolor="#10243a",
        bordercolor="#2dd4bf",
        font=dict(color="#f2f8ff"),
    ),
)


RISK_ORDER = ["Low", "Guarded", "High", "Critical"]
RISK_COLORS = {
    "Low": "#16A34A",
    "Guarded": "#EAB308",
    "High": "#F97316",
    "Critical": "#DC2626",
}
SEVERE_THRESHOLD = 45.0


# ============================================================
# DATA AND THRESHOLDS
# ============================================================

@st.cache_data
def load_dashboard_data():
    required = [PREDICTION_FILE, FEATURE_FILE, METADATA_FILE]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n- " + "\n- ".join(missing)
        )

    predictions = pd.read_csv(
        PREDICTION_FILE,
        parse_dates=["date", "target_start_date", "target_end_date"],
    )
    features = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["date", "target_start_date", "target_end_date"],
    )

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Recreate the exact historical threshold methodology used by risk_engine.py:
    # positive targets from eligible records before 2024-01-01.
    validation_start = pd.Timestamp("2024-01-01")
    historical = features[
        (features["date"] < validation_start)
        & (features["target_end_date"] < validation_start)
        & (features["target_cases_next_2_4w"] > 0)
    ].copy()

    if historical.empty:
        raise ValueError("No eligible positive historical targets for risk thresholds.")

    historical["month"] = historical["date"].dt.month
    cities = sorted(predictions["city"].dropna().unique())

    monthly = (
        historical.groupby(["city", "month"])["target_cases_next_2_4w"]
        .quantile([0.50, 0.80])
        .unstack()
        .reset_index()
    )
    monthly.columns = ["city", "month", "positive_median", "positive_p80"]

    city_thresholds = (
        historical.groupby("city")["target_cases_next_2_4w"]
        .quantile([0.50, 0.80])
        .unstack()
    )
    city_thresholds.columns = ["city_median", "city_p80"]

    national_median = float(historical["target_cases_next_2_4w"].quantile(0.50))
    national_p80 = float(historical["target_cases_next_2_4w"].quantile(0.80))

    threshold_grid = pd.MultiIndex.from_product(
        [cities, range(1, 13)], names=["city", "month"]
    ).to_frame(index=False)

    thresholds = (
        threshold_grid
        .merge(monthly, on=["city", "month"], how="left")
        .merge(city_thresholds, on="city", how="left")
    )
    thresholds["threshold_source"] = "city_month"

    missing_month = thresholds["positive_median"].isna()
    thresholds.loc[missing_month, "positive_median"] = thresholds.loc[
        missing_month, "city_median"
    ]
    thresholds.loc[missing_month, "positive_p80"] = thresholds.loc[
        missing_month, "city_p80"
    ]
    thresholds.loc[missing_month, "threshold_source"] = "city_wide_fallback"

    missing_city = thresholds["positive_median"].isna()
    thresholds.loc[missing_city, "positive_median"] = national_median
    thresholds.loc[missing_city, "positive_p80"] = national_p80
    thresholds.loc[missing_city, "threshold_source"] = "national_fallback"

    thresholds = thresholds.drop(columns=["city_median", "city_p80"])

    data = predictions.copy()
    data["month"] = data["date"].dt.month
    data = data.merge(
        thresholds,
        on=["city", "month"],
        how="left",
        validate="many_to_one",
    )

    if data[["positive_median", "positive_p80"]].isna().any().any():
        raise ValueError("Some predictions have no historical risk thresholds.")

    def assign_risk(value, median, p80):
        if value < 0.5:
            return "Low"
        if value < median:
            return "Guarded"
        if value < p80:
            return "High"
        return "Critical"

    data["predicted_risk"] = [
        assign_risk(v, m, p)
        for v, m, p in zip(
            data["Q50"], data["positive_median"], data["positive_p80"]
        )
    ]
    data["actual_risk"] = [
        assign_risk(v, m, p)
        for v, m, p in zip(
            data["actual"], data["positive_median"], data["positive_p80"]
        )
    ]
    data["absolute_error"] = (data["Q50"] - data["actual"]).abs()
    data["prediction_error"] = data["Q50"] - data["actual"]
    data["month_name"] = data["date"].dt.strftime("%b")
    data["high_warning_label"] = np.where(
        data["high_event_warning"].astype(bool), "Warning", "No warning"
    )
    data["severe_actual"] = data["actual"] >= SEVERE_THRESHOLD

    weather_columns = [
        "date", "city", "ward_id", "rainfall_mm", "avg_temp_c",
        "max_temp_c", "min_temp_c", "relative_humidity_pct",
        "water_logging_risk_index", "sanitation_score",
        "population_density_sqkm", "rainfall_rolling_4w_sum",
    ]
    available = [c for c in weather_columns if c in features.columns]
    weather = features[available].copy()

    data = data.merge(
        weather,
        on=["date", "city", "ward_id"],
        how="left",
        validate="one_to_one",
    )
    return data, thresholds, metadata


def metric_regression(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R²": r2_score(actual, predicted),
        "Bias": np.mean(np.asarray(predicted) - np.asarray(actual)),
    }


def fmt(value, digits=2):
    return "N/A" if pd.isna(value) else f"{value:,.{digits}f}"


# ============================================================
# LOAD
# ============================================================

st.markdown(
    """
    <div class="dg-hero">
        <div class="dg-kicker">AI-powered public health intelligence</div>
        <div class="dg-hero-title">DengueGuard <span style="color:#5eead4">AI</span></div>
        <div class="dg-hero-sub">
            Ward-level dengue forecasting and early-warning intelligence.
            Explore risk patterns, compare city trends, and inspect model predictions
            through an interactive public-health dashboard.
        </div>
        <span class="dg-chip">Q50 FORECAST ENGINE</span>
        <span class="dg-chip">WARD-LEVEL INSIGHTS</span>
        <span class="dg-chip">2025 RETROSPECTIVE EVALUATION</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Forecast target: cumulative dengue cases in weeks t+2, t+3 and t+4. "
    "This dashboard presents retrospective 2025 evaluation data."
)

try:
    data, thresholds, metadata = load_dashboard_data()
except Exception as exc:
    st.error(f"Unable to load dashboard data: {exc}")
    st.info(
        "Run Experiment P first and ensure the processed feature dataset "
        "and generated model/report files are in their expected folders."
    )
    st.stop()

# Verify saved artifacts; dashboard charts use the saved final evaluation predictions.
forecast_artifact_exists = FORECAST_MODEL_FILE.exists()
warning_artifact_exists = WARNING_MODEL_FILE.exists()

with st.sidebar:
    st.title("Explore Dashboard")
    st.caption("Choose a location and risk level to navigate the data.")

    cities = sorted(data["city"].dropna().unique().tolist())
    city_choice = st.selectbox("City", ["All cities"] + cities, index=0)
    city_filtered = (
        data.copy() if city_choice == "All cities"
        else data[data["city"] == city_choice].copy()
    )

    wards = sorted(city_filtered["ward_id"].dropna().unique().tolist())
    ward_choice = st.selectbox("Ward", ["All wards"] + wards, index=0)

    risk_options = [r for r in RISK_ORDER if r in data["predicted_risk"].dropna().unique()]
    risk_choice = st.selectbox("Risk level", ["All risk levels"] + risk_options, index=0)

    min_date = data["date"].min().date()
    max_date = data["date"].max().date()
    selected_dates = st.date_input(
        "Prediction date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.divider()
    st.caption("DengueGuard AI | Hackathon prototype")

filtered = data.copy()
if city_choice != "All cities":
    filtered = filtered[filtered["city"] == city_choice]
if ward_choice != "All wards":
    filtered = filtered[filtered["ward_id"] == ward_choice]
if risk_choice != "All risk levels":
    filtered = filtered[filtered["predicted_risk"] == risk_choice]

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates

filtered = filtered[
    (filtered["date"].dt.date >= start_date)
    & (filtered["date"].dt.date <= end_date)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# ============================================================
# MODEL STATUS AND OVERVIEW
# ============================================================

st.header("Final Model Status")
status1, status2, status3 = st.columns(3)
status1.metric("Forecast model", "Q50")
status2.metric("Forecast artifact", "Available" if forecast_artifact_exists else "Missing")
status3.metric("Warning artifact", "Available" if warning_artifact_exists else "Missing")

st.caption(
    "The displayed 2025 predictions were generated and evaluated by Experiment P. "
    "This dashboard is retrospective; it does not fetch live surveillance or weather data."
)

st.divider()
st.header("Risk Overview")

risk_counts = (
    filtered["predicted_risk"].value_counts().reindex(RISK_ORDER, fill_value=0)
)
high_count = int(risk_counts["High"] + risk_counts["Critical"])
warning_count = int(filtered["high_event_warning"].astype(bool).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Wards monitored", f"{filtered['ward_id'].nunique():,}")
c2.metric("Prediction records", f"{len(filtered):,}")
c3.metric("Low risk", f"{risk_counts['Low']:,}")
c4.metric("Guarded risk", f"{risk_counts['Guarded']:,}")
c5.metric("High + Critical", f"{high_count:,}")

st.caption(
    f"Separate classifier warnings in selected records: {warning_count:,}. "
    "These are ward-date records, not unique wards."
)

with st.expander("✨ Quick data explorer", expanded=False):
    explorer_col1, explorer_col2 = st.columns([1, 2])
    with explorer_col1:
        explore_metric = st.selectbox(
            "Choose a measure",
            ["Q50 forecast", "Actual cases", "High-event probability", "Rainfall"],
            key="quick_explorer_metric",
        )
        explore_group = st.radio(
            "Group results by",
            ["City", "Ward"],
            horizontal=True,
            key="quick_explorer_group",
        )

    metric_map = {
        "Q50 forecast": "Q50",
        "Actual cases": "actual",
        "High-event probability": "high_event_probability",
        "Rainfall": "rainfall_mm",
    }
    chosen_metric = metric_map[explore_metric]
    if chosen_metric not in filtered.columns:
        st.info("This measure is not available in the current data.")
    else:
        group_column = "city" if explore_group == "City" else "ward_id"
        explorer_data = (
            filtered.groupby(group_column, as_index=False)[chosen_metric]
            .mean()
            .sort_values(chosen_metric, ascending=False)
        )
        with explorer_col2:
            fig = px.bar(
                explorer_data,
                x=group_column,
                y=chosen_metric,
                color=chosen_metric,
                color_continuous_scale=["#164e63", "#2dd4bf", "#f59e0b"],
                title=f"Average {explore_metric.lower()} by {explore_group.lower()}",
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="quick_explorer_chart",
                config={"displayModeBar": True, "scrollZoom": True},
            )

# ============================================================
# WARNING PANEL
# ============================================================

st.divider()
st.header("High-Incidence Warning Signals")
warning_share = 100 * filtered["high_event_warning"].astype(bool).mean()
mean_probability = 100 * filtered["high_event_probability"].mean()

w1, w2, w3 = st.columns(3)
w1.metric("Classifier warning records", f"{warning_count:,}")
w2.metric("Warning rate", f"{warning_share:.1f}%")
w3.metric("Mean event probability", f"{mean_probability:.1f}%")

st.info(
    "The classifier is recall-priority: on the 2025 holdout it detected "
    "87.55% of high-incidence events, but precision was 44.23%. "
    "Treat warnings as prompts for review, not confirmed outbreaks."
)

warning_city = (
    filtered.groupby("city", as_index=False)
    .agg(
        warning_records=("high_event_warning", "sum"),
        mean_probability=("high_event_probability", "mean"),
    )
)
wc1, wc2 = st.columns(2)
with wc1:
    fig = px.bar(
        warning_city, x="city", y="warning_records",
        title="Classifier Warning Records by City",
        labels={"warning_records": "Warning records", "city": "City"},
    )
    st.plotly_chart(fig, use_container_width=True)
with wc2:
    fig = px.histogram(
        filtered, x="high_event_probability", nbins=30,
        title="High-Incidence Probability Distribution",
        labels={"high_event_probability": "Predicted probability"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# RISK DISTRIBUTION AND MATRIX
# ============================================================

st.divider()
st.header("Predicted Risk Distribution")

distribution = (
    filtered["predicted_risk"].value_counts()
    .reindex(RISK_ORDER, fill_value=0)
    .rename_axis("Risk Level")
    .reset_index(name="Records")
)
d1, d2 = st.columns(2)
with d1:
    fig = px.bar(
        distribution, x="Risk Level", y="Records",
        color="Risk Level", color_discrete_map=RISK_COLORS,
        category_orders={"Risk Level": RISK_ORDER},
        title="Ward-Date Records by Risk Tier", text="Records",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with d2:
    fig = px.pie(
        distribution, names="Risk Level", values="Records",
        color="Risk Level", color_discrete_map=RISK_COLORS,
        category_orders={"Risk Level": RISK_ORDER},
        hole=0.45, title="Risk Tier Share",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.header("Ward-Level Risk Matrix")
matrix_data = filtered.copy()
matrix_data["risk_number"] = matrix_data["predicted_risk"].map(
    {"Low": 0, "Guarded": 1, "High": 2, "Critical": 3}
)
matrix = matrix_data.pivot_table(
    index="ward_id", columns="date", values="risk_number", aggfunc="first"
)
if not matrix.empty:
    fig = px.imshow(
        matrix, aspect="auto",
        color_continuous_scale=[
            [0.00, RISK_COLORS["Low"]],
            [0.33, RISK_COLORS["Guarded"]],
            [0.66, RISK_COLORS["High"]],
            [1.00, RISK_COLORS["Critical"]],
        ],
        zmin=0, zmax=3,
        labels={"x": "Prediction date", "y": "Ward", "color": "Risk tier"},
        title="Q50 Ward Risk Timeline",
    )
    fig.update_layout(height=max(450, min(1000, 18 * len(matrix))))
    fig.update_coloraxes(
        colorbar=dict(
            tickmode="array", tickvals=[0, 1, 2, 3], ticktext=RISK_ORDER
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CITY ANALYSIS
# ============================================================

st.divider()
st.header("City-Wise Analysis")
city_summary = (
    filtered.groupby("city", as_index=False)
    .agg(
        records=("ward_id", "size"),
        wards=("ward_id", "nunique"),
        average_predicted_cases=("Q50", "mean"),
        average_actual_cases=("actual", "mean"),
        warning_records=("high_event_warning", "sum"),
    )
)
city_risk = pd.crosstab(filtered["city"], filtered["predicted_risk"])
city_risk = city_risk.reindex(columns=RISK_ORDER, fill_value=0).reset_index()

cc1, cc2 = st.columns(2)
with cc1:
    fig = px.bar(
        city_risk, x="city", y=RISK_ORDER, barmode="stack",
        color_discrete_map=RISK_COLORS, title="Risk Tiers Across Cities",
    )
    st.plotly_chart(fig, use_container_width=True)
with cc2:
    fig = px.bar(
        city_summary, x="city", y="average_predicted_cases",
        title="Average Q50 Forecast by City", text_auto=".2f",
    )
    st.plotly_chart(fig, use_container_width=True)
st.dataframe(city_summary.round(2), use_container_width=True, hide_index=True)

# ============================================================
# PERFORMANCE
# ============================================================

st.divider()
st.header("2025 Forecast Performance")

actual = data["actual"].to_numpy()
model_a = data["model_A"].to_numpy()
q50 = data["Q50"].to_numpy()
blend = data["blend_50_50"].to_numpy()

performance = pd.DataFrame([
    {"Model": "Model A", **metric_regression(actual, model_a)},
    {"Model": "Q50 (selected)", **metric_regression(actual, q50)},
    {"Model": "50/50 blend", **metric_regression(actual, blend)},
])
st.dataframe(performance.round(4), use_container_width=True, hide_index=True)

selected_metrics = metric_regression(filtered["actual"], filtered["Q50"])
m1, m2, m3, m4 = st.columns(4)
m1.metric("Filtered MAE", fmt(selected_metrics["MAE"]))
m2.metric("Filtered RMSE", fmt(selected_metrics["RMSE"]))
m3.metric("Filtered R²", fmt(selected_metrics["R²"], 3))
m4.metric("Filtered bias", fmt(selected_metrics["Bias"]))

p1, p2 = st.columns(2)
with p1:
    fig = px.scatter(
        filtered, x="actual", y="Q50", color="predicted_risk",
        color_discrete_map=RISK_COLORS,
        hover_data=["city", "ward_id", "date"],
        title="Q50 Forecast vs Actual Cases",
        labels={"actual": "Actual cases", "Q50": "Q50 predicted cases"},
    )
    max_value = float(max(filtered["actual"].max(), filtered["Q50"].max()))
    fig.add_trace(go.Scatter(
        x=[0, max_value], y=[0, max_value], mode="lines",
        name="Perfect prediction", line=dict(dash="dash"),
    ))
    st.plotly_chart(fig, use_container_width=True)
with p2:
    fig = px.histogram(
        filtered, x="prediction_error", nbins=40,
        title="Q50 Prediction Error Distribution",
        labels={"prediction_error": "Predicted minus actual"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SEVERE EVENT LIMITATION
# ============================================================

st.divider()
st.header("Severe-Event Evaluation")
severe = data[data["actual"] >= SEVERE_THRESHOLD].copy()
severe_count = len(severe)
st.warning(
    f"The 2025 holdout contained only {severe_count} severe records "
    f"(actual target ≥ {SEVERE_THRESHOLD:.0f}). Q50 detected "
    f"{int(((severe['Q50'] >= SEVERE_THRESHOLD)).sum())} of them using the "
    "same numeric threshold. This sample is too small to establish reliable "
    "severe-outbreak detection."
)
if severe_count:
    severe_summary = pd.DataFrame([
        {
            "Model": "Model A",
            "Severe-event MAE": mean_absolute_error(severe["actual"], severe["model_A"]),
            "Underprediction rate": (severe["model_A"] < severe["actual"]).mean(),
        },
        {
            "Model": "Q50",
            "Severe-event MAE": mean_absolute_error(severe["actual"], severe["Q50"]),
            "Underprediction rate": (severe["Q50"] < severe["actual"]).mean(),
        },
        {
            "Model": "50/50 blend",
            "Severe-event MAE": mean_absolute_error(severe["actual"], severe["blend_50_50"]),
            "Underprediction rate": (severe["blend_50_50"] < severe["actual"]).mean(),
        },
    ])
    st.dataframe(severe_summary.round(4), use_container_width=True, hide_index=True)

# ============================================================
# WEATHER AND TABLE
# ============================================================

st.divider()
st.header("Weather and Environmental Insights")
weather_columns = [
    "rainfall_mm", "avg_temp_c", "relative_humidity_pct",
    "water_logging_risk_index", "sanitation_score",
]
available_weather = [c for c in weather_columns if c in filtered.columns]
if available_weather:
    labels = {
        "rainfall_mm": "Average rainfall (mm)",
        "avg_temp_c": "Average temperature (°C)",
        "relative_humidity_pct": "Relative humidity (%)",
        "water_logging_risk_index": "Waterlogging risk index",
        "sanitation_score": "Sanitation score",
    }
    cols = st.columns(len(available_weather))
    for i, col in enumerate(available_weather):
        with cols[i]:
            value = filtered[col].mean()
            st.metric(labels.get(col, col), fmt(value))
    if "rainfall_mm" in filtered:
        trend = filtered.groupby("date", as_index=False)["rainfall_mm"].mean()
        fig = px.line(
            trend, x="date", y="rainfall_mm", markers=True,
            title="Average Weekly Rainfall Across Selected Wards",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Weather values are historical observations associated with the forecast "
        "origin date. These patterns are associations, not proof of causation."
    )

st.divider()
st.header("Ward-Level Prediction Details")
display_columns = [
    "date", "target_start_date", "target_end_date", "city", "ward_id",
    "actual", "Q50", "model_A", "blend_50_50",
    "predicted_risk", "positive_median", "positive_p80",
    "high_event_probability", "high_event_warning",
    "absolute_error", "rainfall_mm", "avg_temp_c",
    "relative_humidity_pct",
]
display_columns = [c for c in display_columns if c in filtered.columns]
table = filtered[display_columns].copy()
st.dataframe(
    table.sort_values("Q50", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download filtered predictions as CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="dengueguard_q50_filtered_predictions.csv",
    mime="text/csv",
)

with st.expander("Historical risk threshold methodology"):
    st.write(
        "Risk tiers are calculated from positive dengue target counts in "
        "eligible 2016–2023 historical records. City-month median and 80th "
        "percentile thresholds are preferred; city-wide and national fallbacks "
        "are used when a group has no positive historical observations."
    )
    st.dataframe(thresholds, use_container_width=True, hide_index=True)

with st.expander("Model methodology and limitations"):
    st.markdown(
        """
        **Forecasting**
        - Selected model: HistGradientBoosting quantile regression (Q50).
        - Features: 46 engineered ward-level features.
        - Forecast target: sum of dengue cases at weeks t+2, t+3 and t+4.
        - 2025 holdout results: MAE 2.6069, RMSE 4.7471, R² 0.8060.

        **High-incidence warning**
        - Separate Random Forest classifier.
        - Training-derived high-event threshold: 27 cases.
        - Probability cutoff: 0.50.
        - 2025 precision: 44.23%; recall: 87.55%; F1: 0.5877.

        **Risk tiers**
        - Low: predicted cases below 0.5.
        - Guarded: positive prediction below historical positive median.
        - High: median to below historical positive 80th percentile.
        - Critical: at or above historical positive 80th percentile.

        **Limitations**
        - This is a retrospective 2025 evaluation dashboard, not a live surveillance feed.
        - Risk tiers are historical relative categories, not official government alert levels.
        - Severe-event sample size in 2025 is very small.
        - The classifier's high recall comes with many false alarms.
        - This prototype is not a substitute for epidemiological or clinical assessment.
        """
    )

st.warning(
    "Research prototype: outputs are model estimates for retrospective evaluation. "
    "They are not official public-health alerts or a substitute for professional "
    "epidemiological assessment."
)
st.markdown(
    '<div class="dg-footer">DengueGuard AI · AI Engineering Hackathon Prototype · '
    'Decision support research, not an official public-health alert system</div>',
    unsafe_allow_html=True,
)
