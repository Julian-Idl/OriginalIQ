from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ai_classifier import roberta_ai_probability
from app.ensemble import FEATURE_ORDER
from app.perplexity import perplexity
from app.stylometry import extract_stylometric_features


def extract_features(text: str) -> dict[str, float]:
    style = extract_stylometric_features(text)
    return {
        **style,
        "roberta_score": roberta_ai_probability(text),
        "perplexity": perplexity(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the logistic-regression ensemble.")
    parser.add_argument("--data", default="data/processed/ai_human.csv", help="CSV with text,label columns.")
    parser.add_argument("--output", default="models/ensemble.joblib")
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("CSV must include text,label columns.")

    feature_rows = []
    for text in frame["text"].astype(str):
        features = extract_features(text)
        feature_rows.append([features.get(name, 0.0) for name in FEATURE_ORDER])

    x_train, x_test, y_train, y_test = train_test_split(
        feature_rows,
        frame["label"].astype(int).tolist(),
        test_size=0.2,
        random_state=42,
        stratify=frame["label"].astype(int).tolist(),
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)


if __name__ == "__main__":
    main()

