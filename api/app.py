from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Heart Disease Prediction API")

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model path
MODEL_PATH = BASE_DIR / "model" / "random_forest_pipeline.pkl"

# Load pipeline
model = joblib.load(MODEL_PATH)

# Feature order MUST match training data
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    # Convert JSON → DataFrame (IMPORTANT)
    input_df = pd.DataFrame([data], columns=FEATURE_ORDER)

    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][prediction]

    return {
        "prediction": int(prediction),
        "confidence": float(confidence)
    }
