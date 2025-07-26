from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class TrainConfig:
    data_path: str
    text_cols: List[str] = field(default_factory=lambda: ["title", "text"])
    target_col: str = "label"
    test_size: float = 0.2
    random_state: int = 42
    ngram_range: tuple = (1, 2)
    max_features: int = 200000
    model: str = "lr"  # lr | lsvc | svm | rf | xgb
    use_class_weight: bool = True
    min_df: int = 3
    lowercase: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    save_dir: str = "artifacts"

@dataclass
class EvalConfig:
    data_path: str
    model_path: str
    vectorizer_path: Optional[str] = None
    label_encoder_path: Optional[str] = None
    text_cols: List[str] = field(default_factory=lambda: ["title", "text"])
    target_col: str = "label"
