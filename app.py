import streamlit as st
import numpy as np
import joblib

# Load saved model & scaler
model = joblib.load("svm_predictive_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔧 Predictive Maintenance System")
st.write("Predict machine failure using sensor data")

# User input fields
air_temp = st.number_input("Air Temperature [K]", min_value=250)
process_temp = st.number_input("Process Temperature [K]", min_value=250)
rot_speed = st.number_input("Rotational Speed [rpm]", min_value=0)
torque = st.number_input("Torque [Nm]", min_value=0.0)
tool_wear = st.number_input("Tool Wear [min]", min_value=0)

# Predict button
if st.button("Predict Machine Condition"):
    input_data = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Machine Failure Detected")
    else:
        st.success("✅ Machine is Operating Normally")
