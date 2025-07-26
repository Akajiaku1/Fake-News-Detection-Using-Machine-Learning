import argparse
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, roc_auc_score
)

from src.config import TrainConfig
from src.data import load_data
from src.features import clean_text, merge_text_columns
from src.models import get_model
from src.utils import ensure_dir, save_json

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, type=str)
    p.add_argument("--model", default="lr", type=str)
    p.add_argument("--save_dir", default="artifacts", type=str)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = TrainConfig(data_path=args.data_path, model=args.model, save_dir=args.save_dir)
    ensure_dir(cfg.save_dir)

    # Load
    df = load_data(cfg.data_path, cfg.text_cols, cfg.target_col)
    df["merged_text"] = df.apply(lambda r: merge_text_columns(r, cfg.text_cols), axis=1)
    df["clean_text"] = df["merged_text"].apply(lambda t: clean_text(
        t,
        lowercase=cfg.lowercase,
        rm_stop=cfg.remove_stopwords,
        lemmatize=cfg.lemmatize
    ))

    X = df["clean_text"].values
    y = df[cfg.target_col].values

    # Encode labels (even though it's binary already, keep consistent)
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        min_df=cfg.min_df
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_valid_vec = vectorizer.transform(X_valid)

    clf = get_model(cfg.model, use_class_weight=cfg.use_class_weight)
    clf.fit(X_train_vec, y_train)

    # Evaluate
    y_prob = None
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_valid_vec)[:, 1]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_valid_vec)
        # map scores to [0,1] via sigmoid fallback
        y_prob = 1 / (1 + np.exp(-scores))
    else:
        y_prob = None

    y_pred = clf.predict(X_valid_vec)
    acc = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average="weighted")
    try:
        roc = roc_auc_score(y_valid, y_prob) if y_prob is not None else None
    except Exception:
        roc = None

    print("\n=== Validation Report ===")
    print(classification_report(y_valid, y_pred))
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}" if roc else f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Persist
    model_path = os.path.join(cfg.save_dir, "model.joblib")
    vec_path = os.path.join(cfg.save_dir, "vectorizer.joblib")
    le_path = os.path.join(cfg.save_dir, "label_encoder.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vec_path)
    joblib.dump(le, le_path)

    metrics = {"accuracy": acc, "f1_weighted": f1, "roc_auc": roc}
    save_json(metrics, os.path.join(cfg.save_dir, "metrics.json"))

    print(f"\nSaved model to: {model_path}")
    print(f"Saved vectorizer to: {vec_path}")
    print(f"Saved label encoder to: {le_path}")

if __name__ == "__main__":
    main()
