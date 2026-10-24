from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

project_dir = Path(__file__).resolve().parent
model_path = project_dir / 'models' / 'predict_target_pipeline.pkl'
bundle = joblib.load(model_path)


class PredictRequest(BaseModel):
    session_id: str
    client_id: str | None = None
    visit_date: str | None = None
    visit_time: str | None = None
    visit_number: int | None = None
    utm_source: str | None = None
    utm_medium: str | None = None
    utm_campaign: str | None = None
    utm_adcontent: str | None = None
    utm_keyword: str | None = None
    device_category: str | None = None
    device_os: str | None = None
    device_brand: str | None = None
    device_model: str | None = None
    device_screen_resolution: str | None = None
    device_browser: str | None = None
    geo_country: str | None = None
    geo_city: str | None = None


class PredictResponse(BaseModel):
    session_id: str
    prediction: int
    probability: float
    threshold: float


@app.get('/status')
def status():
    return {"status": "ok"}


@app.get('/info')
def info():
    return bundle['metadata']


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    X = pd.DataFrame([request.model_dump()])
    probability = bundle['pipeline'].predict_proba(X)[0, 1]
    threshold = bundle['metadata']['threshold']
    prediction = int(probability >= threshold)
    return {
        'session_id': request.session_id,
        'prediction': prediction,
        'probability': probability,
        'threshold': threshold,
    }
