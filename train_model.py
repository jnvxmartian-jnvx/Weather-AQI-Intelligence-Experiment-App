import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv("realistic_aqi_data.csv")

# Features
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

# Target
y = df["AQI"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")

# Save Model
with open("aqi_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved Successfully!")