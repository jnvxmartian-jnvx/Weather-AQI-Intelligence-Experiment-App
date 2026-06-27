import pandas as pd
import random

data = []

cities = [
    ("Delhi", "Delhi"),
    ("Mumbai", "Maharashtra"),
    ("Bengaluru", "Karnataka"),
    ("Chennai", "Tamil Nadu"),
    ("Kolkata", "West Bengal"),
    ("Hyderabad", "Telangana"),
    ("Jaipur", "Rajasthan"),
    ("Lucknow", "Uttar Pradesh"),
    ("Bhopal", "Madhya Pradesh"),
    ("Patna", "Bihar")
]

for i in range(10000):

    city, state = random.choice(cities)

    avg_temp = round(random.uniform(15, 45), 1)
    humidity = round(random.uniform(30, 95), 1)
    rainfall = round(random.uniform(0, 80), 1)
    wind_speed = round(random.uniform(2, 25), 1)
    pressure = round(random.uniform(990, 1025), 1)
    visibility = round(random.uniform(5, 100), 1)

    # Realistic AQI Formula

    aqi = (
        180
        + 0.8 * humidity
        - 1.5 * rainfall
        - 2.0 * wind_speed
        - 1.2 * visibility
        + 0.5 * avg_temp
        + random.uniform(-15, 15)
    )

    aqi = max(20, min(500, round(aqi)))

    data.append([
        city,
        state,
        avg_temp,
        humidity,
        rainfall,
        wind_speed,
        pressure,
        visibility,
        aqi
    ])

df = pd.DataFrame(
    data,
    columns=[
        "City",
        "State",
        "Avg_Temp",
        "Humidity",
        "Rainfall",
        "Wind_Speed",
        "Pressure",
        "Visibility",
        "AQI"
    ]
)

df.to_csv("realistic_aqi_data.csv", index=False)

print("Dataset Created Successfully!")
print("Rows:", len(df))