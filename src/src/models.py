from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_model(name: str, use_class_weight: bool = True):
    name = name.lower()
    if name == "lr":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced" if use_class_weight else None,
            n_jobs=-1,
        )
    elif name == "lsvc":
        return LinearSVC(
            class_weight="balanced" if use_class_weight else None
        )
    elif name == "svm":
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced" if use_class_weight else None
        )
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            n_jobs=-1,
            class_weight="balanced" if use_class_weight else None,
            random_state=42
        )
    elif name == "xgb":
        return XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model: {name}")
