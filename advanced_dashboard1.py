
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="Weather & AQI Intelligence Center",
    layout="wide")
# ----------------------------------
# LOAD DATA
# ----------------------------------

df = pd.read_csv("cleaned_weather_data.csv")
st.write(df.columns.tolist())

# ----------------------------------
# TITLE
# ----------------------------------

st.title("🌍 Weather & AQI Intelligence Center")

# ----------------------------------
# SIDEBAR FILTERS
# ----------------------------------

st.sidebar.header("Filters")

selected_state = st.sidebar.selectbox(
    "Select State",
    sorted(df["State"].unique())
)

filtered_state_df = df[
    df["State"] == selected_state
]

selected_city = st.sidebar.selectbox(
    "Select City",
    sorted(filtered_state_df["City"].unique())
)

filtered_df = filtered_state_df[
    filtered_state_df["City"] == selected_city
]

# ----------------------------------
# DATASET PREVIEW
# ----------------------------------

st.header("Dataset Preview")

st.dataframe(filtered_df)

# ----------------------------------
# CITY STATISTICS
# ----------------------------------

st.header("City Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Records",
        len(filtered_df)
    )

with col2:
    st.metric(
        "Average AQI",
        round(filtered_df["AQI"].mean(), 2)
    )

with col3:
    st.metric(
        "Average Temp",
        round(filtered_df["Avg_Temp"].mean(), 2)
    )

with col4:
    st.metric(
        "Average Rainfall",
        round(filtered_df["Rainfall"].mean(), 2)
    )

# ----------------------------------
# STATE INFORMATION
# ----------------------------------

st.header("Selected Location")

st.write("State:", selected_state)
st.write("City:", selected_city)
# ----------------------------------
# AQI TREND
# ----------------------------------

st.header("📊 AQI Trend")

fig1, ax1 = plt.subplots(figsize=(10, 4))

ax1.plot(
    filtered_df.index,
    filtered_df["AQI"],
    marker="o"
)

ax1.set_title(f"AQI Trend - {selected_city}")
ax1.set_xlabel("Record Number")
ax1.set_ylabel("AQI")

st.pyplot(fig1)

# ----------------------------------
# TEMPERATURE TREND
# ----------------------------------

st.header("🌡️ Temperature Trend")

fig2, ax2 = plt.subplots(figsize=(10, 4))

ax2.plot(
    filtered_df.index,
    filtered_df["Avg_Temp"],
    marker="o"
)

ax2.set_title(f"Temperature Trend - {selected_city}")
ax2.set_xlabel("Record Number")
ax2.set_ylabel("Average Temperature")

st.pyplot(fig2)

# ----------------------------------
# RAINFALL TREND
# ----------------------------------

st.header("🌧️ Rainfall Trend")

fig3, ax3 = plt.subplots(figsize=(10, 4))

ax3.plot(
    filtered_df.index,
    filtered_df["Rainfall"],
    marker="o"
)

ax3.set_title(f"Rainfall Trend - {selected_city}")
ax3.set_xlabel("Record Number")
ax3.set_ylabel("Rainfall")

st.pyplot(fig3)
# ----------------------------------
# ADVANCED CITY ANALYTICS
# ----------------------------------

st.header("📊 Advanced City Analytics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Maximum AQI",
        round(filtered_df["AQI"].max(), 2)
    )

with col2:
    st.metric(
        "Minimum AQI",
        round(filtered_df["AQI"].min(), 2)
    )

with col3:
    st.metric(
        "Average AQI",
        round(filtered_df["AQI"].mean(), 2)
    )

col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Maximum Temperature",
        round(filtered_df["Avg_Temp"].max(), 2)
    )

with col5:
    st.metric(
        "Minimum Temperature",
        round(filtered_df["Avg_Temp"].min(), 2)
    )

with col6:
    st.metric(
        "Average Humidity",
        round(filtered_df["Humidity"].mean(), 2)
    )
    # ----------------------------------
# TOP POLLUTED CITIES
# ----------------------------------

st.header("🏭 Top 5 Most Polluted Cities")

polluted = (
    df.groupby("City")["AQI"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

st.dataframe(polluted)

# ----------------------------------
# CLEANEST CITIES
# ----------------------------------

st.header("🌿 Top 5 Cleanest Cities")

cleanest = (
    df.groupby("City")["AQI"]
    .mean()
    .sort_values()
    .head(5)
)

st.dataframe(cleanest)