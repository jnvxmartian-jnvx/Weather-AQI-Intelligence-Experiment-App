import pickle

with open("aqi_model.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))
print("Model loaded successfully!")