from fastapi import FastAPI
from pydantic import BaseModel
import os

from src.infer import Predictor

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
PREDICTOR = Predictor(
    model_path=os.path.join(ARTIFACT_DIR, "model.joblib"),
    vectorizer_path=os.path.join(ARTIFACT_DIR, "vectorizer.joblib"),
    label_encoder_path=os.path.join(ARTIFACT_DIR, "label_encoder.joblib"),
)

app = FastAPI(title="Fake News Detection API", version="0.1.0")

class PredictRequest(BaseModel):
    text: str

class PredictManyRequest(BaseModel):
    texts: list[str]

@app.get("/")
def root():
    return {"status": "ok", "message": "Fake News Detection API"}

@app.post("/predict")
def predict(req: PredictRequest):
    labels, proba = PREDICTOR.predict([req.text])
    return {"label": int(labels[0]), "probability_fake": proba[0] if proba else None}

@app.post("/predict_many")
def predict_many(req: PredictManyRequest):
    labels, proba = PREDICTOR.predict(req.texts)
    res = []
    for i, t in enumerate(req.texts):
        res.append({
            "text": t,
            "label": int(labels[i]),
            "probability_fake": proba[i] if proba else None
        })
    return {"results": res}
