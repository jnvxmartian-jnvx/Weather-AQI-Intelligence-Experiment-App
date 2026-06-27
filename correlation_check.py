import pandas as pd

df = pd.read_csv("cleaned_weather_data.csv")

cols = [
    "Avg_Temp",
    "Humidity",
    "Rainfall",
    "Wind_Speed",
    "Pressure",
    "Visibility",
    "AQI"
]

print(df[cols].corr()["AQI"].sort_values(ascending=False))