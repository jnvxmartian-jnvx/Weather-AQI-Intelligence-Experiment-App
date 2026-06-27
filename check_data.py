import pandas as pd

df = pd.read_csv("cleaned_weather_data.csv")

print(df.head())

print("\n")
print(df.describe())

print("\nAQI Categories:")
print(df["AQI"].describe())