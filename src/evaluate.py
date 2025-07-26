import argparse
import joblib
import json

from sklearn.metrics import classification_report, confusion_matrix

from src.config import EvalConfig
from src.data import load_data
from src.features import clean_text, merge_text_columns

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, type=str)
    p.add_argument("--model_path", required=True, type=str)
    p.add_argument("--vectorizer_path", default="artifacts/vectorizer.joblib", type=str)
    p.add_argument("--label_encoder_path", default="artifacts/label_encoder.joblib", type=str)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = EvalConfig(
        data_path=args.data_path,
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        label_encoder_path=args.label_encoder_path
    )

    df = load_data(cfg.data_path, cfg.text_cols, cfg.target_col)
    df["merged_text"] = df.apply(lambda r: merge_text_columns(r, cfg.text_cols), axis=1)
    df["clean_text"] = df["merged_text"].apply(lambda t: clean_text(t))

    X = df["clean_text"].values
    y = df[cfg.target_col].values

    clf = joblib.load(cfg.model_path)
    vectorizer = joblib.load(cfg.vectorizer_path)
    le = joblib.load(cfg.label_encoder_path)

    X_vec = vectorizer.transform(X)
    y_true = le.transform(y)

    y_pred = clf.predict(X_vec)

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    print("\n=== Evaluation Report ===")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:", cm)

    with open("artifacts/eval_report.json", "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": cm}, f, indent=2)

if __name__ == "__main__":
    main()
