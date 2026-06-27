import numpy
from reportlab.pdfgen import canvas
import folium
from streamlit_folium import st_folium
import numpy
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle



# ----------------------------------
# LOAD DATA
# ----------------------------------

df = pd.read_csv("cleaned_weather_data.csv")

# Load Model
with open("aqi_model.pkl", "rb") as f:
    model = pickle.load(f)
# -------------------------
# CITY COORDINATES
# -------------------------

city_locations = {
    "Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "Bengaluru": [12.9716, 77.5946],
    "Chennai": [13.0827, 80.2707],
    "Kolkata": [22.5726, 88.3639],
    "Hyderabad": [17.3850, 78.4867],
    "Jaipur": [26.9124, 75.7873],
    "Lucknow": [26.8467, 80.9462],
    "Bhopal": [23.2599, 77.4126],
    "Patna": [25.5941, 85.1376]
}

# -------------------------
# AQI COLOR FUNCTION
# -------------------------

def get_color(aqi):
    if aqi <= 50:
        return "green"
    elif aqi <= 100:
        return "blue"
    elif aqi <= 200:
        return "orange"
    elif aqi <= 300:
        return "red"
    else:
        return "darkred"



# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="Weather & AQI Intelligence Center",
    layout="wide")
# ----------------------------------
# LOAD DATA
# ----------------------------------



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
# ----------------------------------
# AI AQI PREDICTION
# ----------------------------------

st.header("🤖 AI AQI Prediction")

col1, col2 = st.columns(2)

with col1:
    temp = st.number_input(
        "Average Temperature",
        value=30.0
    )

    humidity = st.number_input(
        "Humidity",
        value=60.0
    )

    rainfall = st.number_input(
        "Rainfall",
        value=5.0
    )

with col2:
    wind_speed = st.number_input(
        "Wind Speed",
        value=10.0
    )

    pressure = st.number_input(
        "Pressure",
        value=1010.0
    )

    visibility = st.number_input(
        "Visibility",
        value=50.0
    )

if st.button("Predict AQI"):

    input_data = pd.DataFrame({
        "Avg_Temp": [temp],
        "Humidity": [humidity],
        "Rainfall": [rainfall],
        "Wind_Speed": [wind_speed],
        "Pressure": [pressure],
        "Visibility": [visibility]
    })

    prediction = model.predict(input_data)

    predicted_aqi = prediction[0]

    st.success(
        f"Predicted AQI: {predicted_aqi:.2f}"
    )

    if predicted_aqi <= 50:
        st.success("Category: Good 🟢")

    elif predicted_aqi <= 100:
        st.info("Category: Satisfactory 🟡")

    elif predicted_aqi <= 200:
        st.warning("Category: Moderate 🟠")

    elif predicted_aqi <= 300:
        st.warning("Category: Poor 🔴")

    else:
        st.error("Category: Severe ⚫")
        st.header("📈 AQI Forecasting (Simple)")
def get_color(aqi):
    if aqi <= 50:
        return "green"
    elif aqi <= 100:
        return "blue"
    elif aqi <= 200:
        return "orange"
    elif aqi <= 300:
        return "red"
    else:
        return "darkred"


# -------------------------
# PDF FUNCTION
# -------------------------

def create_pdf(city, aqi, temp, humidity):

    filename = f"{city}_Report.pdf"

    from reportlab.pdfgen import canvas

    c = canvas.Canvas(filename)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "Weather & AQI Report")

    c.setFont("Helvetica", 12)

    c.drawString(50, 760, f"City: {city}")
    c.drawString(50, 740, f"Average AQI: {aqi:.2f}")
    c.drawString(50, 720, f"Average Temperature: {temp:.2f}")
    c.drawString(50, 700, f"Average Humidity: {humidity:.2f}")

    c.save()

    return filename
# Current AQI (from selected city)
today_aqi = float(filtered_df["AQI"].mean())

# Simple trend logic (you can improve later)
trend = 10  # fixed baseline drift

tomorrow_aqi = today_aqi + trend

st.metric("Today AQI", round(today_aqi, 2))
st.metric("Predicted Tomorrow AQI", round(tomorrow_aqi, 2))

# Category
if tomorrow_aqi <= 50:
    st.success("Tomorrow: Good 🟢")
elif tomorrow_aqi <= 100:
    st.info("Tomorrow: Satisfactory 🟡")
elif tomorrow_aqi <= 200:
    st.warning("Tomorrow: Moderate 🟠")
else:
    st.error("Tomorrow: Poor / Severe 🔴")
      
      
# ----------------------------------
# FEATURE IMPORTANCE
# ----------------------------------

st.header("📈 Feature Importance")

feature_names = [
    "Avg_Temp",
    "Humidity",
    "Rainfall",
    "Wind_Speed",
    "Pressure",
    "Visibility"
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(importance_df)

st.bar_chart(
    importance_df.set_index("Feature")
)
# ----------------------------------
# CORRELATION HEATMAP
# ----------------------------------

st.header("🔥 Correlation Heatmap")

corr_columns = [
    "Avg_Temp",
    "Humidity",
    "Rainfall",
    "Wind_Speed",
    "Pressure",
    "Visibility",
    "AQI"
]

corr_matrix = df[corr_columns].corr()

fig, ax = plt.subplots(figsize=(8, 6))

heatmap = ax.imshow(corr_matrix)

ax.set_xticks(range(len(corr_columns)))
ax.set_yticks(range(len(corr_columns)))

ax.set_xticklabels(corr_columns, rotation=45)
ax.set_yticklabels(corr_columns)

for i in range(len(corr_columns)):
    for j in range(len(corr_columns)):
        ax.text(
            j,
            i,
            f"{corr_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.colorbar(heatmap)

st.pyplot(fig)
csv = filtered_df.to_csv(index=False)

st.download_button(
    label="📥 Download City Data",
    data=csv,
    file_name=f"{selected_city}_weather_aqi.csv",
    mime="text/csv"
)

india_map = folium.Map(
    location=[22.9734, 78.6569],
    zoom_start=5,
    tiles="CartoDB positron"
)

for city, coords in city_locations.items():
    city_aqi = df[df["City"] == city]["AQI"].mean()

    folium.CircleMarker(
        location=coords,
        radius=7,
        popup=f"{city} | AQI: {city_aqi:.1f}",
        color=get_color(city_aqi),
        fill=True,
        fill_opacity=0.7
    ).add_to(india_map)

st_folium(india_map, width=700, height=400)

st.header("📄 PDF Report Generator")

if st.button("Generate PDF Report"):

    pdf_file = create_pdf(
        selected_city,
        filtered_df["AQI"].mean(),
        filtered_df["Avg_Temp"].mean(),
        filtered_df["Humidity"].mean()
    )

    with open(pdf_file, "rb") as file:

        st.download_button(
            label="⬇ Download PDF",
            data=file,
            file_name=pdf_file,
            mime="application/pdf"
        )
# ----------------------------------
# AQI FORECAST
# ----------------------------------

st.header("📈 7-Day AQI Forecast")

if st.button("Generate Forecast"):

    current_aqi = filtered_df["AQI"].mean()

    forecast_days = list(range(1, 8))

    forecast_values = []

    for day in forecast_days:

        future_aqi = current_aqi + (day * numpy.random.uniform(1, 5))

        forecast_values.append(round(future_aqi, 2))

    forecast_df = pd.DataFrame({
        "Day": forecast_days,
        "Forecast_AQI": forecast_values
    })

    st.dataframe(forecast_df)

    st.line_chart(
        forecast_df.set_index("Day")
    )
    