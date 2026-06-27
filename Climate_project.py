# ==========================================
# WEATHER & AQI DATA SCIENCE PROJECT
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# LOAD DATASET
# ==========================================

print("=" * 60)
print("WEATHER & AQI DATA SCIENCE PROJECT")
print("=" * 60)

df = pd.read_csv("cleaned_weather_data.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# ==========================================
# DATA INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("DATA INFORMATION")
print("=" * 60)

print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# AVERAGE TEMPERATURE BY CITY
# ==========================================

city_temp = df.groupby("City")["Avg_Temp"].mean()

plt.figure(figsize=(10, 6))
city_temp.plot(kind="bar")

plt.title("Average Temperature by City")
plt.xlabel("City")
plt.ylabel("Average Temperature (°C)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("average_temperature_by_city.png")
plt.show()

# ==========================================
# AQI DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(df["AQI"], bins=20)

plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("aqi_distribution.png")
plt.show()

# ==========================================
# RAINFALL VS AQI
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Rainfall"],
    df["AQI"]
)

plt.title("Rainfall vs AQI")
plt.xlabel("Rainfall (mm)")
plt.ylabel("AQI")

plt.tight_layout()
plt.savefig("rainfall_vs_aqi.png")
plt.show()

# ==========================================
# CORRELATION HEATMAP
# ==========================================

numeric_df = df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# ==========================================
# MACHINE LEARNING
# ==========================================

print("\n" + "=" * 60)
print("MACHINE LEARNING MODEL")
print("=" * 60)

X = df[
    [
        "Avg_Temp",
        "Humidity",
        "Rainfall",
        "Wind_Speed",
        "Pressure",
        "Visibility"
    ]
]

y = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# ==========================================
# PREDICTION
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nModel Performance")
print("-" * 30)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8, 5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ==========================================
# ACTUAL VS PREDICTED AQI
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")

plt.tight_layout()
plt.savefig("actual_vs_predicted_aqi.png")
plt.show()

# ==========================================
# SAMPLE AQI PREDICTION
# ==========================================

print("\n" + "=" * 60)
print("NEW AQI PREDICTION")
print("=" * 60)

new_data = pd.DataFrame({
    "Avg_Temp": [30],
    "Humidity": [70],
    "Rainfall": [15],
    "Wind_Speed": [8],
    "Pressure": [1012],
    "Visibility": [50]
})

predicted_aqi = model.predict(new_data)

print(f"\nPredicted AQI = {predicted_aqi[0]:.2f}")

# ==========================================
# SAVE PREDICTIONS
# ==========================================

prediction_df = pd.DataFrame({
    "Actual_AQI": y_test.values,
    "Predicted_AQI": predictions
})

prediction_df.to_csv(
    "aqi_predictions.csv",
    index=False
)

print("\nPrediction file saved:")
print("aqi_predictions.csv")

# ==========================================
# TOP 5 MOST POLLUTED CITIES
# ==========================================

print("\n" + "=" * 60)
print("TOP 5 MOST POLLUTED CITIES")
print("=" * 60)

top_polluted = (
    df.groupby("City")["AQI"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

print(top_polluted)

# ==========================================
# TOP 5 CLEANEST CITIES
# ==========================================

print("\n" + "=" * 60)
print("TOP 5 CLEANEST CITIES")
print("=" * 60)

cleanest = (
    df.groupby("City")["AQI"]
    .mean()
    .sort_values()
    .head(5)
)

print(cleanest)

# ==========================================
# PROJECT COMPLETED
# ==========================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")
print("1. average_temperature_by_city.png")
print("2. aqi_distribution.png")
print("3. rainfall_vs_aqi.png")
print("4. correlation_heatmap.png")
print("5. feature_importance.png")
print("6. actual_vs_predicted_aqi.png")
print("7. aqi_predictions.csv")

print("\nWeather & AQI Analysis Complete!")