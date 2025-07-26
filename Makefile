.PHONY: setup train evaluate api streamlit test lint

setup:
	python -m pip install -r requirements.txt
	python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

train:
	python -m src.train --data_path data/raw/train.csv

evaluate:
	python -m src.evaluate --data_path data/raw/train.csv --model_path artifacts/model.joblib

api:
	uvicorn api.app:app --reload

streamlit:
	streamlit run app/streamlit_app.py

test:
	pytest

lint:
	ruff check .
