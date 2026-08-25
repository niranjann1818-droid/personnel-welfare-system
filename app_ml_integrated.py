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
# LOGIN PAGE - VISUAL DESIGN
# ------------------------------------------------------------
if not st.session_state.logged_in:

    # ========================================================
    # LOGIN PAGE DESIGN
    # ========================================================

    st.markdown("""
    <style>
   .stApp p,
.stApp label {
    color: #172033 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
}

[data-testid="stCaptionContainer"] {
    color: #475569 !important;
}

[data-testid="stWidgetLabel"] label {
    color: #172033 !important;
}

    /* Background */
    .stApp {
        background:
            radial-gradient(circle at 10% 20%,
                rgba(37,99,235,0.10), transparent 28%),
            radial-gradient(circle at 90% 80%,
                rgba(14,165,233,0.10), transparent 30%),
            linear-gradient(135deg,#f8fbff,#eef6ff);
    }

    /* Remove unnecessary top space */
    .block-container {
        padding-top: 3rem;
    }

    /* Login card */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 22px !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
        box-shadow: 0 15px 40px rgba(30,64,175,0.10);
        background: rgba(255,255,255,0.82);
    }

    /* Button */
    div.stButton > button {
        border-radius: 12px;
        height: 48px;
        font-weight: 700;
    }

    </style>
    """, unsafe_allow_html=True)


    # ========================================================
    # BRAND HEADER
    # ========================================================

    st.markdown(
        "## 🛡️ TECH ATELIER"
    )

    st.caption(
        "Personnel Welfare Intelligence • AI-Powered Predictive Monitoring"
    )

    st.divider()


    # ========================================================
    # MAIN LOGIN LAYOUT
    # ========================================================

    left, right = st.columns([1.35, 0.85], gap="large")


    # ========================================================
    # LEFT SIDE – PROBLEM STATEMENT
    # ========================================================

    with left:

        st.markdown("### 🧠 Personnel Welfare Intelligence")

        st.markdown(
            """
            #### Protecting personnel through early risk identification

            Personnel working under demanding conditions may experience
            **stress, fatigue, excessive workload and reduced wellbeing**.

            This system uses personnel welfare indicators and an
            **AI-powered XGBoost model** to identify potential burnout-risk
            patterns and support welfare decision-making.
            """
        )

        st.divider()

        f1, f2 = st.columns(2)

        with f1:
            st.info(
                "🧠 **Stress Monitoring**\n\n"
                "Tracks stress-related welfare indicators."
            )

        with f2:
            st.info(
                "💤 **Fatigue Analysis**\n\n"
                "Considers fatigue, sleep and workload patterns."
            )

        f3, f4 = st.columns(2)

        with f3:
            st.info(
                "📊 **Risk Analytics**\n\n"
                "Provides organization-level welfare trends."
            )

        with f4:
            st.info(
                "🤖 **AI Prediction**\n\n"
                "FORCEFLOW XGBoost estimates burnout-risk categories."
            )

        st.success(
            "🛡️ Early Detection  →  Better Welfare  →  Stronger Personnel"
        )


    # ========================================================
    # RIGHT SIDE – LOGIN CARD
    # ========================================================

    with right:

        with st.container(border=True):

            st.markdown("### 🔐 Secure Login")

            st.caption(
                "Authorized welfare personnel only"
            )

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

            st.caption(
                "Privacy-first welfare monitoring • "
                "AI-assisted decision support"
            )


    st.divider()

    st.caption(
        "TECH ATELIER • Personnel Stress & Welfare Monitoring System"
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
# LOAD TRAINED ML MODEL - NEW FORCEFLOW MODEL
# ============================================================

@st.cache_resource
def load_ml_model():
    model = xgb.XGBClassifier()
    model.load_model("forceflow_new_model.json")
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
    "attendance_score",
    "performance_score",
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
    "stress_risk_score",
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
    0: "Critical",
    1: "High",
    2: "Low",
    3: "Moderate",
}

def predict_ml_burnout(dataframe):
    """Return ML burnout-risk prediction and confidence for each row."""

    X = dataframe[ML_FEATURES].copy()

    X["rank_level"] = X["rank_level"].map(RANK_MAP)

    X = X.replace([float("inf"), float("-inf")], 0).fillna(0)

    predictions = ml_model.predict(X).astype(int)
   
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
latest_df["ai_burnout_risk"], latest_df["ai_burnout_confidence"] = predict_ml_burnout(latest_df)


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
# MODERN UI STYLING
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at 90% 10%, rgba(37, 99, 235, 0.08), transparent 25%),
            radial-gradient(circle at 10% 90%, rgba(14, 165, 233, 0.06), transparent 25%),
            #f7faff;
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 700;
    }

    /* Section headings */
    h1, h2, h3 {
        color: #0f172a;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eff6ff 0%, #f8fafc 100%);
        border-right: 1px solid #dbeafe;
    }

    /* Info / success / warning boxes */
    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

</style>
""", unsafe_allow_html=True)
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

    st.caption(
        "Individual personnel welfare and stress-risk analysis"
    )

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
    # PERSONNEL PROFILE
    # ========================================================

    st.subheader(
        f"👤 Personnel Profile — {selected_person}"
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
        "Rank Level",
        latest["rank_level"]
    )

    c4.metric(
        "Week",
        latest["week"]
    )

    st.divider()

    # ========================================================
    # WORK & DUTY INFORMATION
    # ========================================================

    st.subheader("💼 Work & Duty Information")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Duty Hours",
        f"{latest['duty_hours']:.1f}"
    )

    c2.metric(
        "Consecutive Duty Days",
        latest["consecutive_duty_days"]
    )

    c3.metric(
        "Overtime Hours",
        f"{latest['overtime_hours']:.1f}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Workload Score",
        f"{latest['workload_score']:.1f}"
    )

    c2.metric(
        "Sleep Hours",
        f"{latest['sleep_hours']:.1f}"
    )

    c3.metric(
        "Attendance Score",
        f"{latest['attendance_score']:.1f}"
    )

    st.divider()

    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.subheader("📊 Performance Information")

    c1, c2 = st.columns(2)

    c1.metric(
        "Performance Score",
        f"{latest['performance_score']:.1f}"
    )

    c2.metric(
        "Stress Risk Score",
        f"{latest['stress_risk_score']:.1f}"
    )

    st.divider()

    # ========================================================
    # SERVICE & DEPLOYMENT
    # ========================================================

    st.subheader("🛡️ Service & Deployment")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Transfer Frequency",
        latest["transfer_frequency"]
    )

    c2.metric(
        "Training Hours",
        f"{latest['training_hours']:.1f}"
    )

    c3.metric(
        "Deployment Days",
        latest["deployment_days"]
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Deployment Frequency",
        latest["deployment_frequency"]
    )

    c2.metric(
        "Operational Exposure",
        f"{latest['operational_exposure']:.1f}"
    )

    st.divider()

    # ========================================================
    # LEAVE INFORMATION
    # ========================================================

    st.subheader("🌴 Leave Information")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Leave Days",
        latest["leave_days"]
    )

    c2.metric(
        "Days Since Last Leave",
        latest["days_since_last_leave"]
    )

    c3.metric(
        "Leave Frequency",
        latest["leave_frequency"]
    )

    st.divider()

    # ========================================================
    # CURRENT STRESS RISK
    # ========================================================

    st.subheader("⚠️ Current Stress Risk")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Stress Risk",
            latest["stress_risk"]
        )

    with c2:

        st.metric(
            "Stress Risk Score",
            f"{latest['stress_risk_score']:.1f}"
        )

    # ========================================================
    # AI PREDICTION
    # ========================================================

    st.subheader("🤖 AI Predicted Stress Risk")

    ai_risk, ai_confidence = predict_ml_burnout(
        person_data.iloc[[-1]]
    )

    ai_risk_value = ai_risk.iloc[0]

    ai_confidence_value = ai_confidence.iloc[0]

    ai_c1, ai_c2 = st.columns(2)

    with ai_c1:

        st.metric(
            "ML Stress Risk",
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
    # STRESS RISK TREND
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

    st.divider()

    # ========================================================
    # CONTRIBUTING FACTORS
    # ========================================================

    st.subheader(
        "🔎 Current Welfare Indicators"
    )

    indicators = {

        "Duty Hours":
            latest["duty_hours"],

        "Consecutive Duty Days":
            latest["consecutive_duty_days"],

        "Overtime Hours":
            latest["overtime_hours"],

        "Workload Score":
            latest["workload_score"],

        "Sleep Hours":
            latest["sleep_hours"],

        "Attendance Score":
            latest["attendance_score"],

        "Performance Score":
            latest["performance_score"],

        "Stress Risk Score":
            latest["stress_risk_score"],

        "Deployment Days":
            latest["deployment_days"],

        "Operational Exposure":
            latest["operational_exposure"],

        "Leave Days":
            latest["leave_days"],

        "Days Since Last Leave":
            latest["days_since_last_leave"]
    }

    indicator_df = pd.DataFrame(
        list(indicators.items()),
        columns=[
            "Indicator",
            "Value"
        ]
    )

    st.dataframe(
        indicator_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ========================================================
    # WELFARE OBSERVATION
    # ========================================================

    st.subheader("📌 Welfare Observation")

    observations = []

    if latest["workload_score"] >= 7:

        observations.append(
            "High workload"
        )

    if latest["sleep_hours"] < 6:

        observations.append(
            "Reduced sleep duration"
        )

    if latest["overtime_hours"] >= 15:

        observations.append(
            "High overtime"
        )

    if latest["consecutive_duty_days"] >= 10:

        observations.append(
            "Extended consecutive duty"
        )

    if latest["stress_risk_score"] >= 75:

        observations.append(
            "High stress-risk score"
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

    st.subheader(" Suggested Welfare Action")

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
            "maintain healthy workload and rest patterns."
        )

# ============================================================
# ADD NEW PERSONNEL
# ============================================================

elif page == "➕ Add Personnel":

    st.title("➕ Add New Personnel")

    st.caption(
        "Enter personnel welfare information to generate an AI stress-risk prediction."
    )

    st.divider()

    st.info(
        "The information entered here is processed using the trained "
        "FORCEFLOW XGBoost model. This is a welfare-risk prediction "
        "and not a medical diagnosis."
    )

    # ========================================================
    # PERSONNEL INFORMATION
    # ========================================================

    st.subheader("👤 Personnel Information")

    c1, c2, c3 = st.columns(3)

    with c1:

        personnel_id = st.text_input(
            "Personnel ID",
            placeholder="Example: P5001"
        )

        week = st.number_input(
            "Week",
            min_value=1,
            value=1,
            step=1
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=80,
            value=30,
            step=1
        )

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
            [
                "Junior",
                "Intermediate",
                "Senior"
            ]
        )

        duty_hours = st.number_input(
            "Duty Hours",
            min_value=0.0,
            max_value=24.0,
            value=8.0,
            step=0.5
        )

        consecutive_duty_days = st.number_input(
            "Consecutive Duty Days",
            min_value=0,
            value=5,
            step=1
        )

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

        sleep_hours = st.number_input(
            "Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

        attendance_score = st.number_input(
            "Attendance Score",
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=0.1
        )

        performance_score = st.number_input(
            "Performance Score",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1
        )

    st.divider()

    # ========================================================
    # SERVICE & DEPLOYMENT
    # ========================================================

    st.subheader("🛡️ Service & Deployment")

    c1, c2, c3 = st.columns(3)

    with c1:

        transfer_frequency = st.number_input(
            "Transfer Frequency",
            min_value=0,
            value=1,
            step=1
        )

        training_hours = st.number_input(
            "Training Hours",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

        deployment_days = st.number_input(
            "Deployment Days",
            min_value=0,
            value=10,
            step=1
        )

    with c2:

        deployment_frequency = st.number_input(
            "Deployment Frequency",
            min_value=0,
            value=1,
            step=1
        )

        operational_exposure = st.number_input(
            "Operational Exposure",
            min_value=0.0,
            value=5.0,
            step=0.5
        )

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

        stress_risk_score = st.number_input(
            "Stress Risk Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.1
        )

    st.divider()

    # ========================================================
    # PREDICT
    # ========================================================

    if st.button(
        "🤖 Predict Stress Risk",
        type="primary",
        use_container_width=True
    ):

        if not personnel_id.strip():

            st.error(
                "Please enter a Personnel ID."
            )

            st.stop()

        # ====================================================
        # CREATE NEW PERSON RECORD
        # ====================================================

        new_person = pd.DataFrame([{

            "week": week,

            "duty_hours": duty_hours,

            "consecutive_duty_days": consecutive_duty_days,

            "overtime_hours": overtime_hours,

            "workload_score": workload_score,

            "sleep_hours": sleep_hours,

            "attendance_score": attendance_score,

            "performance_score": performance_score,

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

            "leave_frequency": leave_frequency,

            "stress_risk_score": stress_risk_score

        }])

        # ====================================================
        # RUN ML MODEL
        # ====================================================

        prediction, confidence = predict_ml_burnout(
            new_person
        )

        predicted_risk = prediction.iloc[0]

        predicted_confidence = confidence.iloc[0]

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "🤖 AI Stress-Risk Prediction"
        )

        r1, r2 = st.columns(2)

        with r1:

            st.metric(
                "Predicted Stress Risk",
                predicted_risk
            )

        with r2:

            st.metric(
                "Prediction Confidence",
                f"{predicted_confidence:.1f}%"
            )

        st.success(
            f"Personnel {personnel_id} has been assessed "
            f"with an AI stress-risk category of "
            f"**{predicted_risk}**."
        )

        st.caption(
            "This prediction supports welfare decision-making "
            "and is not a medical diagnosis."
        )

        # ====================================================
        # SUBMITTED INFORMATION
        # ====================================================

        st.divider()

        st.subheader(
            "📋 Submitted Personnel Information"
        )

        display_data = new_person.copy()

        display_data.insert(
            0,
            "personnel_id",
            personnel_id
        )

        display_data["ai_stress_risk"] = predicted_risk

        display_data["ai_stress_confidence"] = (
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