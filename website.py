
import streamlit as st
import pandas as pd
import pickle

# Load Model
with open("aqi_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("🌍 AI Powered AQI Prediction System")

st.write("Predict Air Quality Index using weather conditions")

# User Inputs
temp = st.number_input(
    "Average Temperature (°C)",
    min_value=0.0,
    max_value=60.0,
    value=30.0
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    value=10.0
)

wind = st.number_input(
    "Wind Speed",
    min_value=0.0,
    value=8.0
)

pressure = st.number_input(
    "Pressure",
    min_value=900.0,
    max_value=1100.0,
    value=1012.0
)

visibility = st.number_input(
    "Visibility",
    min_value=0.0,
    value=50.0
)

# Prediction Button
if st.button("Predict AQI"):

    input_data = pd.DataFrame({
        "Avg_Temp": [temp],
        "Humidity": [humidity],
        "Rainfall": [rainfall],
        "Wind_Speed": [wind],
        "Pressure": [pressure],
        "Visibility": [visibility]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted AQI: {prediction[0]:.2f}"
    )

    # AQI Category
    aqi = prediction[0]

    if aqi <= 50:
        st.success("AQI Category: Good 🟢")
    elif aqi <= 100:
        st.info("AQI Category: Moderate 🟡")
    elif aqi <= 200:
        st.warning("AQI Category: Poor 🟠")
    else:
        st.error("AQI Category: Very Poor 🔴")





