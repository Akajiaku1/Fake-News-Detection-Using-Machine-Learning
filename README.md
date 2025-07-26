# Fake-News-Detection-Using-Machine-Learning

A clean, reproducible **scikit‑learn** pipeline for fake news detection using **TF‑IDF + classical ML** (Logistic Regression, Linear SVM, Random Forest, XGBoost). Comes with:

- End-to-end training & evaluation CLIs
- FastAPI inference microservice
- Streamlit interactive demo
- Unit tests
- Dockerfile

## 1. Quickstart

```bash
git clone https://github.com/Akajiaku1/Fake-News-Detection-Using-Machine-Learning.git
cd Fake-News-Detection-Using-Machine-Learning

python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
pip install -r requirements.txt
make setup

Place your dataset at data/raw/train.csv. Expecting Kaggle format with columns:
[id, title, author, text, label] where label ∈ {0 (real), 1 (fake)}.
Train

python -m src.train --data_path data/raw/train.csv --model lr

Evaluate

python -m src.evaluate --data_path data/raw/train.csv --model_path artifacts/model.joblib

Serve API

uvicorn api.app:app --reload
# POST http://127.0.0.1:8000/predict {"text": "Breaking: ..." }

Streamlit UI

streamlit run app/streamlit_app.py

Docker

docker build -t fake-news-ml .
docker run -p 8000:8000 fake-news-ml

2. Models supported

    lr (default): LogisticRegression

    lsvc: LinearSVC

    svm: SVC (rbf)

    rf: RandomForestClassifier

    xgb: XGBClassifier

3. Project layout

(see repository tree in this README’s header)
4. Reproducibility

    All artifacts are saved to artifacts/:

        model.joblib – trained model

        vectorizer.joblib – TF-IDF vectorizer

        label_encoder.joblib – label encoder

        metrics.json – metrics summary

5. License

MIT


---

### `LICENSE` (MIT – optional)
```txt
MIT License

Copyright (c) ...

Permission is hereby granted, free of charge, to any person obtaining a copy...
