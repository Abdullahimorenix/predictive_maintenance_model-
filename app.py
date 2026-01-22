import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Load model and scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# App layout
# -----------------------------
st.set_page_config(page_title="Predictive Maintenance", layout="centered")

st.title("🛠️ Predictive Maintenance System")
st.write("Predict machine failure based on sensor readings.")

# -----------------------------
# User inputs
# -----------------------------
st.sidebar.header("Input Machine Parameters")

air_temp = st.sidebar.number_input("Air temperature [K]", min_value=250.0, max_value=350.0, value=300.0)
process_temp = st.sidebar.number_input("Process temperature [K]", min_value=250.0, max_value=350.0, value=310.0)
rot_speed = st.sidebar.number_input("Rotational speed [rpm]", min_value=1000, max_value=3000, value=1500)
torque = st.sidebar.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0)
tool_wear = st.sidebar.number_input("Tool wear [min]", min_value=0, max_value=300, value=50)

# -----------------------------
# Prepare input
# -----------------------------
input_data = pd.DataFrame([[
    air_temp,
    process_temp,
    rot_speed,
    torque,
    tool_wear
]], columns=[
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
])

input_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Machine Condition"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Machine Failure Likely")
    else:
        st.success("✅ Machine Operating Normally")

# -----------------------------
# Model info
# -----------------------------
with st.expander("ℹ️ Model Information"):
    st.write("""
    **Model:** Support Vector Machine (RBF Kernel)  
    **Preprocessing:** StandardScaler  
    **Output:** Binary classification  
    """)

