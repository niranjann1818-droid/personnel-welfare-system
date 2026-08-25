import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import xgboost as xgb
import os
from pathlib import Path

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Personnel Welfare System",
    page_icon="🛡️",
    layout="wide"
)
# ============================================================
# LOGIN SYSTEM
# ============================================================

# Demo credentials for SIH prototype
USERS = {
    "admin": "admin123",
    "welfare": "welfare123"
}

# Initialize login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# ------------------------------------------------------------
# LOGIN PAGE
# ------------------------------------------------------------

if not st.session_state.logged_in:

    st.markdown(
        """
        <div style="text-align:center; padding-top:60px;">
            <h1>🛡️ Personnel Welfare System</h1>
            <h3>TECH ATELIER</h3>
            <p>AI-Powered Predictive Welfare Monitoring</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Center the login box
    left, center, right = st.columns([1, 1.5, 1])

    with center:

        st.subheader("🔐 Secure Login")

        username = st.text_input(
            "Username",
            placeholder="Enter username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password"
        )

        if st.button(
            "🔐 Login",
            type="primary",
            use_container_width=True
        ):

            if username in USERS and USERS[username] == password:

                st.session_state.logged_in = True
                st.session_state.username = username

                st.success("Login successful!")

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    st.stop()

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
    print("DEBUG PREDICTION:", predictions)
    print("DEBUG MODEL CLASSES:", ml_model.classes_)

    probabilities = ml_model.predict_proba(X)

    confidence = probabilities.max(axis=1) * 100

    risk_labels = pd.Series(
        predictions,
        index=dataframe.index
    ).map(ML_RISK_MAP)

    return (
        risk_labels,
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

logo_path = Path(__file__).parent / "tech_atelier_logo.jpeg.jpeg"

if logo_path.exists():
    st.sidebar.image(str(logo_path), width = 180)

st.sidebar.title("🛡️ Welfare System")
st.sidebar.caption("SIH26186")


st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "👤 Personnel Analysis",
        "➕ Add Personnel",
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

        st.subheader("Burnout Risk Distribution")

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
# ADD NEW PERSONNEL
# ============================================================

elif page == "➕ Add Personnel":

    st.title("➕ Add New Personnel")
    st.caption(
        "Enter personnel welfare information to generate an AI burnout-risk prediction."
    )

    st.divider()

    st.info(
        "The information entered here is processed using the trained "
        "FORCEFLOW XGBoost model. This is a welfare-risk prediction "
        "and not a medical diagnosis."
    )

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Personnel Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        personnel_id = st.text_input(
            "Personnel ID",
            placeholder="Example: P5001"
        )

    with c2:
        week = st.number_input(
            "Week",
            min_value=1,
            value=1,
            step=1
        )

    with c3:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=80,
            value=30,
            step=1
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        service_years = st.number_input(
            "Service Years",
            min_value=0,
            max_value=50,
            value=5,
            step=1
        )

    with c2:
        rank_level = st.selectbox(
            "Rank Level",
            ["Junior", "Intermediate", "Senior"]
        )

    with c3:
        duty_hours = st.number_input(
            "Duty Hours",
            min_value=0.0,
            max_value=24.0,
            value=8.0,
            step=0.5
        )

    st.divider()

    # --------------------------------------------------------
    # WORK & FATIGUE
    # --------------------------------------------------------

    st.subheader("💼 Work & Fatigue")

    c1, c2, c3 = st.columns(3)

    with c1:
        consecutive_duty_days = st.number_input(
            "Consecutive Duty Days",
            min_value=0,
            value=5,
            step=1
        )

    with c2:
        overtime_hours = st.number_input(
            "Overtime Hours",
            min_value=0.0,
            value=5.0,
            step=0.5
        )

    with c3:
        workload_score = st.slider(
            "Workload Score",
            0.0,
            10.0,
            5.0,
            0.1
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        sleep_hours = st.number_input(
            "Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

    with c2:
        fatigue_score = st.slider(
            "Fatigue Score",
            0.0,
            10.0,
            5.0,
            0.1
        )

    with c3:
        stress_score = st.slider(
            "Stress Score",
            0.0,
            10.0,
            5.0,
            0.1
        )

    st.divider()

    # --------------------------------------------------------
    # WELLBEING
    # --------------------------------------------------------

    st.subheader("🧠 Wellbeing Indicators")

    c1, c2, c3 = st.columns(3)

    with c1:
        mood_score = st.slider(
            "Mood Score",
            0.0,
            10.0,
            5.0,
            0.1
        )

    with c2:
        emotional_exhaustion = st.slider(
            "Emotional Exhaustion",
            0.0,
            10.0,
            5.0,
            0.1
        )

    with c3:
        attendance_score = st.slider(
            "Attendance Score",
            0.0,
            10.0,
            8.0,
            0.1
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        activity_level = st.slider(
            "Activity Level",
            0.0,
            10.0,
            5.0,
            0.1
        )

    with c2:
        performance_score = st.slider(
            "Performance Score",
            0.0,
            10.0,
            7.0,
            0.1
        )

    with c3:
        social_withdrawal = st.slider(
            "Social Withdrawal",
            0.0,
            10.0,
            3.0,
            0.1
        )

    c1, c2 = st.columns(2)

    with c1:
        behavioral_change = st.slider(
            "Behavioral Change",
            0.0,
            10.0,
            3.0,
            0.1
        )

    with c2:
        transfer_frequency = st.number_input(
            "Transfer Frequency",
            min_value=0,
            value=1,
            step=1
        )

    st.divider()

    # --------------------------------------------------------
    # SERVICE & DEPLOYMENT
    # --------------------------------------------------------

    st.subheader("🛡️ Service & Deployment")

    c1, c2, c3 = st.columns(3)

    with c1:
        training_hours = st.number_input(
            "Training Hours",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

    with c2:
        deployment_days = st.number_input(
            "Deployment Days",
            min_value=0,
            value=10,
            step=1
        )

    with c3:
        deployment_frequency = st.number_input(
            "Deployment Frequency",
            min_value=0,
            value=1,
            step=1
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        operational_exposure = st.number_input(
            "Operational Exposure",
            min_value=0.0,
            value=5.0,
            step=0.5
        )

    with c2:
        leave_days = st.number_input(
            "Leave Days",
            min_value=0,
            value=10,
            step=1
        )

    with c3:
        days_since_last_leave = st.number_input(
            "Days Since Last Leave",
            min_value=0,
            value=30,
            step=1
        )

    leave_frequency = st.number_input(
        "Leave Frequency",
        min_value=0,
        value=2,
        step=1
    )

    st.divider()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🤖 Predict Burnout Risk",
        type="primary",
        use_container_width=True
    ):

        if not personnel_id.strip():
            st.error("Please enter a Personnel ID.")
            st.stop()

        # Create one row using the EXACT model feature names.
        new_person = pd.DataFrame([{
            "week": week,
            "duty_hours": duty_hours,
            "consecutive_duty_days": consecutive_duty_days,
            "overtime_hours": overtime_hours,
            "workload_score": workload_score,
            "sleep_hours": sleep_hours,
            "fatigue_score": fatigue_score,
            "stress_score": stress_score,
            "mood_score": mood_score,
            "emotional_exhaustion": emotional_exhaustion,
            "attendance_score": attendance_score,
            "activity_level": activity_level,
            "performance_score": performance_score,
            "social_withdrawal": social_withdrawal,
            "behavioral_change": behavioral_change,
            "age": age,
            "service_years": service_years,
            "rank_level": rank_level,
            "transfer_frequency": transfer_frequency,
            "training_hours": training_hours,
            "deployment_days": deployment_days,
            "deployment_frequency": deployment_frequency,
            "operational_exposure": operational_exposure,
            "leave_days": leave_days,
            "days_since_last_leave": days_since_last_leave,
            "leave_frequency": leave_frequency
        }])

        # Run the trained ML model.
        prediction, confidence = predict_ml_burnout(
            new_person
        )

        predicted_risk = prediction.iloc[0]
        predicted_confidence = confidence.iloc[0]

        st.divider()

        st.subheader("🤖 AI Prediction Result")

        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                "Predicted Burnout Risk",
                predicted_risk
            )

        with r2:
            st.metric(
                "Prediction Confidence",
                f"{predicted_confidence:.1f}%"
            )

        st.success(
            f"Personnel {personnel_id} has been assessed "
            f"with an AI burnout-risk category of "
            f"**{predicted_risk}**."
        )

        st.caption(
            "This prediction supports welfare decision-making "
            "and is not a medical diagnosis."
        )

        st.divider()

        st.subheader("📋 Submitted Personnel Information")

        display_data = new_person.copy()
        display_data.insert(
            0,
            "personnel_id",
            personnel_id
        )

        display_data["ai_burnout_risk"] = predicted_risk
        display_data["ai_burnout_confidence"] = (
            predicted_confidence
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
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