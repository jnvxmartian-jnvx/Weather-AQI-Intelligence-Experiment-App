import streamlit as st
import pandas as pd
import pickle

# ----------------------
# PAGE CONFIG
# ----------------------

st.set_page_config(
    page_title="Weather & AQI Intelligence Center",
    layout="wide"
)

# ----------------------
# LOAD DATA
# ----------------------

df = pd.read_csv("cleaned_weather_data.csv")

# ----------------------
# LOAD MODEL
# ----------------------

with open("aqi_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------
# SIDEBAR
# ----------------------

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Data Explorer",
        "Analysis",
        "AQI Prediction",
        "Project Report"
    ]
)