import re
import string
from typing import Iterable
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

URL_RE = re.compile(r"http\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
PUNC_TABLE = str.maketrans("", "", string.punctuation)

_stopwords = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

def clean_text(
    text: str,
    lowercase: bool = True,
    rm_stop: bool = True,
    lemmatize: bool = True
) -> str:
    if lowercase:
        text = text.lower()

    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = text.translate(PUNC_TABLE)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    if rm_stop:
        tokens = [t for t in tokens if t not in _stopwords and len(t) > 2]

    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

def merge_text_columns(row, text_cols: Iterable[str]) -> str:
    return " ".join([str(row[c]) for c in text_cols])
