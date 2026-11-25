
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="NYC 311 Service Analytics",
    page_icon="🏙️",
    layout="wide"
)

DB_PATH = Path("data/nyc_311.db")


@st.cache_data
def load_data():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("""
            SELECT
                created_month,
                borough,
                agency,
                problem_formerly_complaint_type AS complaint,
                status,
                resolution_hours
            FROM service_requests
        """, conn)


df = load_data()

st.title("🏙️ NYC 311 Public Service Analytics")
st.caption(
    "296,434 service requests | Four sampled weeks in 2025"
)

# SIDEBAR FILTERS
st.sidebar.header("Filters")

months = sorted(df["created_month"].dropna().unique())
boroughs = sorted(df["borough"].dropna().unique())

selected_months = st.sidebar.multiselect(
    "Sample month",
    months,
    default=months
)

selected_boroughs = st.sidebar.multiselect(
    "Borough",
    boroughs,
    default=boroughs
)

filtered = df[
    df["created_month"].isin(selected_months)
    & df["borough"].isin(selected_boroughs)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# Valid closed cases for resolution analysis
valid = filtered[
    filtered["status"].astype("string")
    .str.strip().str.lower().eq("closed")
    & filtered["resolution_hours"].notna()
    & filtered["resolution_hours"].ge(0)
].copy()

valid["delayed_7d"] = valid["resolution_hours"] > 168

# KPI METRICS
total = len(filtered)
closed = len(valid)

median_hours = (
    valid["resolution_hours"].median()
    if closed else 0
)

delay_pct = (
    valid["delayed_7d"].mean() * 100
    if closed else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Requests", f"{total:,}")
c2.metric("Valid Closed Requests", f"{closed:,}")
c3.metric("Median Resolution", f"{median_hours:.2f} hrs")
c4.metric("Over 7 Days", f"{delay_pct:.2f}%")

st.divider()


tab1, tab2, tab3, tab4 = st.tabs([
    "Complaint Overview",
    "Resolution Performance",
    "Seasonal Trends",
    "Machine Learning"
])

# TAB 1: COMPLAINT VOLUME
with tab1:
    st.subheader("Requests by Borough")

    borough_counts = (
        filtered["borough"]
        .value_counts()
        .rename_axis("borough")
        .reset_index(name="requests")
    )

    st.bar_chart(
        borough_counts,
        x="borough",
        y="requests"
    )

    st.subheader("Top 10 Complaint Categories")

    top_complaints = (
        filtered["complaint"]
        .value_counts()
        .head(10)
        .rename_axis("complaint")
        .reset_index(name="requests")
    )

    st.bar_chart(
        top_complaints,
        x="complaint",
        y="requests",
        horizontal=True
    )

# TAB 2: RESOLUTION PERFORMANCE
with tab2:
    st.subheader("Resolution Performance by Borough")

    if valid.empty:
        st.info("No valid closed requests found.")
    else:
        performance = (
            valid.groupby("borough")
            .agg(
                closed_requests=("delayed_7d", "size"),
                delay_rate=("delayed_7d", "mean"),
                median_hours=("resolution_hours", "median"),
                avg_hours=("resolution_hours", "mean")
            )
            .reset_index()
        )

        performance["delay_pct"] = (
            performance["delay_rate"] * 100
        )

        st.bar_chart(
            performance,
            x="borough",
            y="delay_pct",
            horizontal=True
        )

        st.dataframe(
            performance[
                [
                    "borough",
                    "closed_requests",
                    "median_hours",
                    "avg_hours",
                    "delay_pct"
                ]
            ].round(2),
            use_container_width=True,
            hide_index=True
        )

# TAB 3: SEASONAL ANALYSIS
with tab3:
    st.subheader("Heating Complaint Seasonality")

    seasonal = filtered.assign(
        heating=filtered["complaint"].eq("HEAT/HOT WATER")
    )

    heating = (
        seasonal.groupby("created_month")["heating"]
        .agg(["size", "sum"])
        .reset_index()
    )

    heating.columns = [
        "created_month",
        "total_requests",
        "heating_requests"
    ]

    heating["heating_pct"] = (
        heating["heating_requests"]
        / heating["total_requests"] * 100
    )

    st.bar_chart(
        heating,
        x="created_month",
        y="heating_pct"
    )

    st.dataframe(
        heating.round(2),
        use_container_width=True,
        hide_index=True
    )

# TAB 4: MACHINE-LEARNING PERFORMANCE
with tab4:
    st.subheader("Predicting 7-Day Closure Delays")

    st.info(
        "Retrospective evaluation using October 2025 "
        "requests that eventually closed. This is not "
        "a validated real-time forecasting system."
    )

    results_path = Path(
        "reports/tables/ml_holdout_evaluation.csv"
    )

    calibration_path = Path(
        "reports/tables/model_calibration.csv"
    )

    missed_path = Path(
        "reports/tables/missed_delays.csv"
    )

    false_path = Path(
        "reports/tables/false_alarms.csv"
    )

    if results_path.exists():
        results = pd.read_csv(results_path)
        metrics = results.iloc[0]

        a, b, c, d = st.columns(4)

        a.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        b.metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.4f}"
        )

        c.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )

        d.metric(
            "Average Precision",
            f"{metrics['pr_auc']:.4f}"
        )

    st.divider()

    if calibration_path.exists():
        st.subheader("Prediction Calibration")

        calibration = pd.read_csv(calibration_path)

        st.bar_chart(
            calibration,
            x="risk_group",
            y=[
                "actual_delay_pct",
                "predicted_delay_pct"
            ]
        )

        st.caption(
            "Compares average predicted risk against "
            "actual delay frequency within each risk band."
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Most Common Missed Delays")

        if missed_path.exists():
            missed = pd.read_csv(
                missed_path,
                usecols=["complaint"]
            )

            st.bar_chart(
                missed["complaint"].value_counts().head(10)
            )

    with right:
        st.subheader("Most Common False Alarms")

        if false_path.exists():
            false_alarms = pd.read_csv(
                false_path,
                usecols=["complaint"]
            )

            st.bar_chart(
                false_alarms["complaint"].value_counts().head(10)
            )

    st.caption(
        "Error charts show counts rather than error "
        "rates. Frequent complaint types may have "
        "more errors simply because they are common."
    )

st.divider()

st.caption(
    "Source: NYC Open Data 311 Service Requests. "
    "Resolution metrics include only closed requests "
    "with valid nonnegative durations. "
    "Samples do not represent full calendar months. "
    "Closure does not necessarily mean a problem was resolved."
)
