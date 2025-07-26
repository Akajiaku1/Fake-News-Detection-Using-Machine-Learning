import pandas as pd
from typing import List

def load_data(path: str, text_cols: List[str], target_col: str):
    df = pd.read_csv(path)
    # Basic sanity checks
    missing_cols = set(text_cols + [target_col]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    # Fill NA
    for c in text_cols:
        df[c] = df[c].fillna("")

    df[target_col] = df[target_col].astype(int)
    return df
