import pandas as pd
from sklearn.model_selection import train_test_split
from src.features import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from src.models import get_model

def test_train_pipeline_end2end(tmp_path):
    # Small synthetic dataset
    data = {
        "title": ["A", "B", "C", "D"],
        "text":  ["Real news text", "Fake news text", "Another real", "Another fake"],
        "label": [0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    df["merged_text"] = df["title"] + " " + df["text"]
    df["clean_text"] = df["merged_text"].apply(clean_text)

    X = df["clean_text"].values
    y = df["label"].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X_train_vec = vec.fit_transform(X_train)
    X_valid_vec = vec.transform(X_valid)

    model = get_model("lr")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_valid_vec)
    assert len(preds) == len(y_valid)
