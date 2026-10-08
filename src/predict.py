import joblib
import pandas as pd
import os

# Get the project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model path
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "human_activity_model.pkl"
)

# Load trained model
model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

# Example sensor data
new_data = pd.DataFrame({
    "accX": [-0.50],
    "accY": [3.78],
    "accZ": [8.90],
    "gyroX": [0.02],
    "gyroY": [0.01],
    "gyroZ": [0.03]
})

# Prediction
prediction = model.predict(new_data)[0]

# Probability
probability = model.predict_proba(new_data)[0]

print("Prediction:", prediction)
print("Probability:", probability)