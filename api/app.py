from fastapi import FastAPI, Request
import joblib
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-api")

app = FastAPI(title="Heart Disease Prediction API")

# Resolve base directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "random_forest_pipeline.pkl"

model = joblib.load(MODEL_PATH)

# Simple in-memory metric
REQUEST_COUNT = 0


@app.middleware("http")
async def log_requests(request: Request, call_next):
    global REQUEST_COUNT
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT += 1
    logger.info(
        f"Method={request.method} "
        f"Path={request.url.path} "
        f"Status={response.status_code} "
        f"Time={duration:.3f}s "
        f"TotalRequests={REQUEST_COUNT}"
    )

    return response


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.post("/predict")
def predict(data: dict):
    # Convert input JSON to DataFrame with column names
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    confidence = model.predict_proba(df)[0][prediction]

    return {
        "prediction": int(prediction),
        "confidence": float(confidence)
    }


@app.get("/metrics")
def metrics():
    return {
        "total_requests": REQUEST_COUNT
    }
