import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import xgboost as xgb

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Personnel Welfare System",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("final_dataset.csv")



df = load_data()

# ============================================================
# LOAD TRAINED ML MODEL
# ============================================================

@st.cache_resource
def load_ml_model():
    model = xgb.XGBClassifier()
    model.load_model("forceflow_model.json")
    return model
ml_model = load_ml_model()
    # These are the exact 26 features used to train forceflow_model.pkl
ML_FEATURES = [
    "week",
    "duty_hours",
    "consecutive_duty_days",
    "overtime_hours",
    "workload_score",
    "sleep_hours",
    "fatigue_score",
    "stress_score",
    "mood_score",
    "emotional_exhaustion",
    "attendance_score",
    "activity_level",
    "performance_score",
    "social_withdrawal",
    "behavioral_change",
    "age",
    "service_years",
    "rank_level",
    "transfer_frequency",
    "training_hours",
    "deployment_days",
    "deployment_frequency",
    "operational_exposure",
    "leave_days",
    "days_since_last_leave",
    "leave_frequency",
]

# rank_level was label-encoded alphabetically during model training.
RANK_MAP = {
    "Intermediate": 0,
    "Junior": 1,
    "Senior": 2,
}

# The trained model contains 3 classes.
# Based on the notebook's class mapping and the dataset labels:
# 0 = High, 1 = Low, 2 = Moderate.
ML_RISK_MAP = {
    0: "High",
    1: "Low",
    2: "Moderate",
}

def predict_ml_burnout(dataframe):
    """Return ML burnout-risk prediction and confidence for each row."""
    X = dataframe[ML_FEATURES].copy()
    X["rank_level"] = X["rank_level"].map(RANK_MAP)
    X = X.replace([float("inf"), float("-inf")], 0).fillna(0)

    predictions = ml_model.predict(X).astype(int)
    probabilities = ml_model.predict_proba(X)
    confidence = probabilities.max(axis=1) * 100

    return (
        pd.Series(predictions, index=dataframe.index).map(ML_RISK_MAP),
        pd.Series(confidence, index=dataframe.index),
    )

# Latest record for every personnel
latest_df = (
    df.sort_values("week")
    .groupby("personnel_id")
    .tail(1)
    .reset_index(drop=True)
)

# ML prediction for the latest record of each personnel
latest_df["ai_burnout_risk"], latest_df["ai_burnout_confidence"] = (
    predict_ml_burnout(latest_df)
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ Welfare System")
st.sidebar.caption("SIH26186")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "👤 Personnel Analysis",
        "📊 Risk Analytics",
        "⚠️ Alerts",
        "💡 Recommendations"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Privacy-first welfare monitoring system "
    "designed for authorized welfare personnel."
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🛡️ Personnel Stress & Welfare Monitoring System")

    st.caption(
        "AI-powered predictive personnel welfare monitoring platform"
    )

    st.divider()

    # Risk counts
    low_count = (
        latest_df["stress_risk"] == "Low"
    ).sum()

    moderate_count = (
        latest_df["stress_risk"] == "Moderate"
    ).sum()

    high_count = (
        latest_df["stress_risk"] == "High"
    ).sum()

    critical_count = (
        latest_df["stress_risk"] == "Critical"
    ).sum()

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Total Personnel",
        f"{len(latest_df):,}"
    )

    c2.metric(
        "🟢 Low",
        f"{low_count:,}"
    )

    c3.metric(
        "🟡 Moderate",
        f"{moderate_count:,}"
    )

    c4.metric(
        "🔴 High",
        f"{high_count:,}"
    )

    c5.metric(
        "⚫ Critical",
        f"{critical_count:,}"
    )

    st.divider()

    # AI burnout summary
    ai_high_count = (latest_df["ai_burnout_risk"] == "High").sum()
    ai_moderate_count = (latest_df["ai_burnout_risk"] == "Moderate").sum()
    ai_low_count = (latest_df["ai_burnout_risk"] == "Low").sum()

    st.subheader("🤖 AI Burnout Prediction Summary")

    a1, a2, a3 = st.columns(3)
    a1.metric("AI Low", f"{ai_low_count:,}")
    a2.metric("AI Moderate", f"{ai_moderate_count:,}")
    a3.metric("AI High", f"{ai_high_count:,}")

    st.caption(
        "These values are predictions from the trained FORCEFLOW XGBoost model."
    )

    st.divider()

    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    left, right = st.columns(2)

    with left:

        st.subheader("📊 Stress Risk Distribution")

        stress_counts = (
            latest_df["stress_risk"]
            .value_counts()
            .reset_index()
        )

        stress_counts.columns = [
            "Risk Level",
            "Personnel"
        ]

        fig = px.pie(
            stress_counts,
            names="Risk Level",
            values="Personnel",
            hole=0.45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("🔥 Burnout Risk Distribution")

        burnout_counts = (
            latest_df["burnout_risk"]
            .value_counts()
            .reset_index()
        )

        burnout_counts.columns = [
            "Risk Level",
            "Personnel"
        ]

        fig2 = px.pie(
            burnout_counts,
            names="Risk Level",
            values="Personnel",
            hole=0.45
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.divider()

    # ========================================================
    # PERSONNEL REQUIRING ATTENTION
    # ========================================================

    st.subheader("⚠️ Personnel Requiring Attention")

    attention = latest_df[
        latest_df["stress_risk"].isin(
            ["High", "Critical"]
        )
    ]

    attention_columns = [
        "personnel_id",
        "week",
        "stress_risk",
        "burnout_risk",
        "workload_score",
        "stress_score",
        "fatigue_score",
        "deployment_days"
    ]

    st.dataframe(
        attention[attention_columns].head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PERSONNEL ANALYSIS
# ============================================================

elif page == "👤 Personnel Analysis":

    st.title("👤 Personnel Analysis")

    personnel_list = sorted(
        df["personnel_id"].unique()
    )

    selected_person = st.selectbox(
        "Select Personnel ID",
        personnel_list
    )

    person_data = df[
        df["personnel_id"] == selected_person
    ].sort_values("week")

    latest = person_data.iloc[-1]

    st.divider()

    # ========================================================
    # PROFILE
    # ========================================================

    st.subheader(
        f"Personnel Profile — {selected_person}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Age",
        latest["age"]
    )

    c2.metric(
        "Service Years",
        latest["service_years"]
    )

    c3.metric(
        "Deployment Days",
        latest["deployment_days"]
    )

    c4.metric(
        "Workload Score",
        latest["workload_score"]
    )

    st.divider()

    # ========================================================
    # CURRENT WELFARE RISK
    # ========================================================

    st.subheader("Current Welfare Risk")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Stress Risk",
            latest["stress_risk"]
        )

        st.write(
            f"Risk Score: {latest['stress_risk_score']:.1f}"
        )

    with c2:

        st.metric(
            "Burnout Risk",
            latest["burnout_risk"]
        )

        st.write(
            f"Risk Score: {latest['burnout_risk_score']:.1f}"
        )

    # ========================================================
    # AI BURNOUT PREDICTION
    # ========================================================

    st.subheader("🤖 AI Predicted Burnout Risk")

    ai_risk, ai_confidence = predict_ml_burnout(
        person_data.iloc[[-1]]
    )

    ai_risk_value = ai_risk.iloc[0]
    ai_confidence_value = ai_confidence.iloc[0]

    ai_c1, ai_c2 = st.columns(2)

    with ai_c1:
        st.metric(
            "ML Burnout Risk",
            ai_risk_value
        )

    with ai_c2:
        st.metric(
            "Prediction Confidence",
            f"{ai_confidence_value:.1f}%"
        )

    st.caption(
        "XGBoost prediction from the trained FORCEFLOW model. "
        "This supports welfare decision-making and is not a medical diagnosis."
    )

    st.divider()

    # ========================================================
    # STRESS TREND
    # ========================================================

    st.subheader("📈 Stress Risk Trend")

    fig = px.line(
        person_data,
        x="week",
        y="stress_risk_score",
        markers=True,
        labels={
            "week": "Week",
            "stress_risk_score": "Stress Risk Score"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # BURNOUT TREND
    # ========================================================

    st.subheader("📈 Burnout Risk Trend")

    fig2 = px.line(
        person_data,
        x="week",
        y="burnout_risk_score",
        markers=True,
        labels={
            "week": "Week",
            "burnout_risk_score": "Burnout Risk Score"
        }
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    # ========================================================
    # CONTRIBUTING FACTORS
    # ========================================================

    st.subheader("🔎 Factors Contributing to Current Risk")

    indicators = {
        "Workload Score": latest["workload_score"],
        "Stress Score": latest["stress_score"],
        "Fatigue Score": latest["fatigue_score"],
        "Emotional Exhaustion": latest["emotional_exhaustion"],
        "Sleep Hours": latest["sleep_hours"],
        "Overtime Hours": latest["overtime_hours"],
        "Consecutive Duty Days": latest["consecutive_duty_days"],
        "Leave Days": latest["leave_days"],
        "Days Since Last Leave": latest["days_since_last_leave"],
        "Deployment Days": latest["deployment_days"],
        "Social Withdrawal": latest["social_withdrawal"],
        "Behavioral Change": latest["behavioral_change"]
    }

    indicator_df = pd.DataFrame(
        list(indicators.items()),
        columns=["Indicator", "Value"]
    )

    st.dataframe(
        indicator_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # WELFARE OBSERVATION
    # ========================================================

    st.subheader("📌 Welfare Observation")

    observations = []

    if latest["workload_score"] >= 7:
        observations.append("High workload")

    if latest["fatigue_score"] >= 7:
        observations.append("Elevated fatigue")

    if latest["stress_score"] >= 7:
        observations.append("Elevated stress")

    if latest["sleep_hours"] < 6:
        observations.append("Reduced sleep duration")

    if latest["overtime_hours"] >= 15:
        observations.append("High overtime")

    if latest["consecutive_duty_days"] >= 10:
        observations.append(
            "Extended consecutive duty"
        )

    if observations:

        st.warning(
            "Potential contributing indicators: "
            + ", ".join(observations)
            + "."
        )

    else:

        st.info(
            "No major threshold-based contributing "
            "indicators were detected in the current record."
        )

    # ========================================================
    # WELFARE ACTION
    # ========================================================

    st.subheader("💡 Suggested Welfare Action")

    if latest["stress_risk"] == "Critical":

        st.error(
            "Prompt confidential welfare follow-up by "
            "authorized personnel according to "
            "organizational protocols."
        )

    elif latest["stress_risk"] == "High":

        st.warning(
            "Consider a confidential welfare check-in "
            "and review workload, deployment and rest patterns."
        )

    elif latest["stress_risk"] == "Moderate":

        st.info(
            "Consider a voluntary welfare check-in and "
            "continue monitoring workload and rest patterns."
        )

    else:

        st.success(
            "Continue routine welfare monitoring and "
            "healthy work-rest practices."
        )


# ============================================================
# RISK ANALYTICS
# ============================================================

elif page == "📊 Risk Analytics":

    st.title("📊 Risk Analytics")

    st.caption(
        "Organization-level welfare and operational trends"
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Stress Score",
        f"{latest_df['stress_score'].mean():.2f}"
    )

    c2.metric(
        "Average Fatigue",
        f"{latest_df['fatigue_score'].mean():.2f}"
    )

    c3.metric(
        "Average Workload",
        f"{latest_df['workload_score'].mean():.2f}"
    )

    st.divider()

    # Workload vs Stress
    st.subheader("📈 Workload vs Stress")

    fig = px.scatter(
        latest_df,
        x="workload_score",
        y="stress_score",
        hover_name="personnel_id",
        labels={
            "workload_score": "Workload Score",
            "stress_score": "Stress Score"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # Deployment vs Stress
    st.subheader("📈 Deployment vs Stress Risk")

    fig2 = px.scatter(
        latest_df,
        x="deployment_days",
        y="stress_risk_score",
        hover_name="personnel_id",
        labels={
            "deployment_days": "Deployment Days",
            "stress_risk_score": "Stress Risk Score"
        }
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    # ========================================================
    # AI BURNOUT PREDICTION DISTRIBUTION
    # ========================================================

    st.subheader("🤖 AI Burnout Risk Distribution")

    ai_counts = (
        latest_df["ai_burnout_risk"]
        .value_counts()
        .reindex(["Low", "Moderate", "High"], fill_value=0)
        .reset_index()
    )

    ai_counts.columns = ["Risk Level", "Personnel"]

    fig3 = px.pie(
        ai_counts,
        names="Risk Level",
        values="Personnel",
        hole=0.45
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.caption(
        "AI predictions are generated by the trained XGBoost FORCEFLOW model. "
        "The current trained model predicts Low, Moderate and High burnout risk; "
        "Critical is not a trained ML class in this model."
    )


# ============================================================
# ALERTS
# ============================================================

elif page == "⚠️ Alerts":

    st.title("⚠️ Welfare Alerts")

    st.caption(
        "Alerts for authorized welfare personnel"
    )

    st.divider()

    critical = latest_df[
        latest_df["stress_risk"] == "Critical"
    ]

    high = latest_df[
        latest_df["stress_risk"] == "High"
    ]

    if len(critical) > 0:

        st.error(
            f"⚫ {len(critical)} personnel identified "
            "with Critical stress risk."
        )

        st.dataframe(
            critical[
                [
                    "personnel_id",
                    "stress_risk_score",
                    "burnout_risk",
                    "workload_score",
                    "stress_score",
                    "fatigue_score"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    if len(high) > 0:

        st.warning(
            f"🔴 {len(high)} personnel identified "
            "with High stress risk."
        )

        st.dataframe(
            high[
                [
                    "personnel_id",
                    "stress_risk_score",
                    "burnout_risk",
                    "workload_score",
                    "stress_score",
                    "fatigue_score"
                ]
            ].head(50),
            use_container_width=True,
            hide_index=True
        )

    if len(critical) == 0 and len(high) == 0:

        st.success(
            "No High or Critical stress-risk "
            "personnel detected."
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "💡 Recommendations":

    st.title("💡 Welfare Recommendations")

    st.write(
        "Recommendations support welfare intervention "
        "and are not disciplinary decisions."
    )

    st.divider()

    st.subheader("🟢 Low Risk")

    st.write(
        "Continue routine welfare monitoring and "
        "encourage healthy work-rest practices."
    )

    st.subheader("🟡 Moderate Risk")

    st.write(
        "Consider a voluntary welfare check-in, "
        "review workload patterns and continue monitoring."
    )

    st.subheader("🔴 High Risk")

    st.write(
        "Recommend confidential welfare follow-up "
        "and review workload, deployment and leave patterns."
    )

    st.subheader("⚫ Critical Risk")

    st.write(
        "Prompt authorized welfare personnel for "
        "confidential and timely intervention according "
        "to organizational welfare protocols."
    )

    st.divider()

    st.info(
        "This prototype supports welfare decisions. "
        "It does not diagnose mental health conditions "
        "or make disciplinary decisions."
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "SIH26186 | Personnel Welfare Monitoring Prototype"
)