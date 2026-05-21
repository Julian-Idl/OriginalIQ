from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .config import get_settings

FEATURE_ORDER = [
    "roberta_score",
    "perplexity",
    "sentence_length_variance",
    "lexical_diversity",
    "pos_entropy",
    "punctuation_ratio",
    "burstiness",
]

_ensemble = None


def feature_vector(features: dict[str, float]) -> np.ndarray:
    vector = [float(features.get(name, 0.0)) for name in FEATURE_ORDER]
    vector[1] = min(vector[1], 500.0) / 500.0
    return np.asarray(vector, dtype=np.float32).reshape(1, -1)


def get_ensemble():
    global _ensemble
    if _ensemble is not None:
        return _ensemble
    path = Path(get_settings().ensemble_model_path)
    if path.exists():
        _ensemble = joblib.load(path)
    return _ensemble


def predict_ai_score(features: dict[str, float]) -> float:
    model = get_ensemble()
    if model is None:
        roberta = features.get("roberta_score", 0.5)
        ppl = min(features.get("perplexity", 80.0), 500.0)
        perplexity_signal = max(0.0, min(1.0, 1.0 - (ppl / 500.0)))
        stylometry_signal = 0.5
        return 0.65 * roberta + 0.25 * perplexity_signal + 0.10 * stylometry_signal

    probabilities = model.predict_proba(feature_vector(features))[0]
    return float(probabilities[1])


def explain_ai(features: dict[str, float], score: float) -> str:
    ppl = features.get("perplexity", 0.0)
    diversity = features.get("lexical_diversity", 0.0)
    variance = features.get("sentence_length_variance", 0.0)

    reasons = []
    if features.get("roberta_score", 0.0) >= 0.65:
        reasons.append("the transformer classifier found AI-like phrasing")
    if ppl and ppl < 45:
        reasons.append("GPT-2 perplexity is low, indicating highly predictable wording")
    elif ppl > 120:
        reasons.append("perplexity is higher, which weighs toward human variation")
    if variance < 20:
        reasons.append("sentence lengths are unusually uniform")
    if diversity < 0.38:
        reasons.append("lexical diversity is limited")

    if not reasons:
        reasons.append("signals are mixed across classifier, perplexity, and stylometry")

    label = "AI-like" if score >= 0.5 else "human-like"
    return f"The ensemble rates the passage as {label} because " + "; ".join(reasons) + "."

