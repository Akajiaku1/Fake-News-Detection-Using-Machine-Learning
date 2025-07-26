import joblib
from typing import List

from src.features import clean_text

class Predictor:
    def __init__(self, model_path: str, vectorizer_path: str, label_encoder_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.le = joblib.load(label_encoder_path)

    def predict(self, texts: List[str]):
        cleaned = [clean_text(t) for t in texts]
        X_vec = self.vectorizer.transform(cleaned)
        preds = self.model.predict(X_vec)
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_vec)[:, 1].tolist()
        return self.le.inverse_transform(preds).tolist(), proba
