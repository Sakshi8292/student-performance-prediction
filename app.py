"""
app.py
-------
Streamlit web app for the Student Performance Prediction project.

Run locally:
    streamlit run app.py

Deploy live (FREE):
    1. Push this repo to GitHub
    2. Go to https://share.streamlit.io
    3. Sign in with GitHub -> "New app" -> select this repo -> main file: app.py -> Deploy
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

MODEL_PATH = "models/linear_regression_model.pkl"
SCALER_PATH = "models/scaler.pkl"

FEATURES = [
    "hours_studied",
    "attendance_pct",
    "previous_scores",
    "sleep_hours",
    "extracurricular",
    "parental_education",
    "internet_access",
]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


st.title("🎓 Student Performance Predictor")
st.markdown(
    "Predict a student's **final exam score** based on study habits, attendance, "
    "and background — powered by a **Linear Regression** model trained with "
    "Scikit-Learn."
)

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run `python src/generate_dataset.py`, "
        "`python src/data_cleaning.py`, and `python src/train_model.py` first."
    )
    st.stop()

st.divider()
st.subheader("📝 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.slider("Hours studied per day", 0.0, 12.0, 5.0, 0.5)
    attendance_pct = st.slider("Attendance (%)", 30.0, 100.0, 80.0, 1.0)
    previous_scores = st.slider("Previous exam score", 0.0, 100.0, 65.0, 1.0)
    sleep_hours = st.slider("Average sleep (hours/night)", 3.0, 10.0, 6.5, 0.5)

with col2:
    extracurricular = st.selectbox("Extracurricular activities?", ["Yes", "No"])
    parental_education = st.selectbox(
        "Parental education", ["High School", "Bachelors", "Masters", "PhD"]
    )
    internet_access = st.selectbox("Internet access at home?", ["Yes", "No"])

edu_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}

input_dict = {
    "hours_studied": hours_studied,
    "attendance_pct": attendance_pct,
    "previous_scores": previous_scores,
    "sleep_hours": sleep_hours,
    "extracurricular": 1 if extracurricular == "Yes" else 0,
    "parental_education": edu_map[parental_education],
    "internet_access": 1 if internet_access == "Yes" else 0,
}

input_df = pd.DataFrame([input_dict])[FEATURES]

st.divider()

if st.button("🔮 Predict Final Score", type="primary", use_container_width=True):
    X_scaled = scaler.transform(input_df)
    prediction = model.predict(X_scaled)[0]
    prediction = float(np.clip(prediction, 0, 100))

    st.subheader("📊 Result")
    st.metric("Predicted Final Score", f"{prediction:.1f} / 100")

    if prediction >= 75:
        st.success("Excellent! This student is predicted to perform very well. 🌟")
    elif prediction >= 50:
        st.info("Average performance predicted. There's room for improvement. 📈")
    else:
        st.warning("This student may need additional academic support. 📚")

st.divider()
with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        **Project**: Student Performance Prediction (Beginner ML Project)

        **Tech stack**: Pandas, NumPy, Scikit-Learn, Streamlit

        **Pipeline**:
        1. Synthetic dataset generation (`generate_dataset.py`)
        2. Data cleaning — missing values, duplicates, outliers (`data_cleaning.py`)
        3. Linear Regression model training & evaluation (`train_model.py`)
        4. Interactive web app for live predictions (`app.py`)

        Built as a learning project to understand the **end-to-end ML workflow**.
        """
    )
