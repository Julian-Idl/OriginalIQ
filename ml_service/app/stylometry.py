from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np

from .preprocessing import get_nlp
from .similarity import tokenize


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def extract_stylometric_features(text: str) -> dict[str, float]:
    nlp = get_nlp()
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents if sent.text.strip()]
    sentence_lengths = [len(tokenize(sentence)) for sentence in sentences]
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    pos_tags = [token.pos_ for token in doc if not token.is_space]
    punctuation = len(re.findall(r"[^\w\s]", text))
    chars = max(len(text), 1)

    lexical_diversity = len(set(tokens)) / len(tokens) if tokens else 0.0
    return {
        "sentence_length_mean": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        "sentence_length_variance": float(np.var(sentence_lengths)) if sentence_lengths else 0.0,
        "lexical_diversity": lexical_diversity,
        "pos_entropy": entropy(pos_tags),
        "punctuation_ratio": punctuation / chars,
        "burstiness": float(np.std(sentence_lengths)) if sentence_lengths else 0.0,
    }

