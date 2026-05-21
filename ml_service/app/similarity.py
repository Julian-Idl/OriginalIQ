from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def ngram_overlap(a: str, b: str, sizes: tuple[int, ...] = (3, 4, 5)) -> float:
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    scores = []
    for size in sizes:
        scores.append(jaccard(ngrams(tokens_a, size), ngrams(tokens_b, size)))
    return float(sum(scores) / len(scores)) if scores else 0.0


def combined_similarity(
    cosine: float,
    overlap: float,
    cross_score: float | None = None,
    lcs_signal: float = 0.0,
) -> float:
    cross = 1 / (1 + math.exp(-cross_score)) if cross_score is not None else cosine
    score = 0.25 * cosine + 0.20 * cross + 0.30 * overlap + 0.25 * lcs_signal
    return max(0.0, min(1.0, score))


def lexical_contribution(matches: list[dict]) -> list[dict]:
    totals = Counter()
    for match in matches:
        url = match.get("source_url") or "local-corpus"
        totals[url] += match.get("score", 0.0)
    total_score = sum(totals.values()) or 1.0
    return [
        {"url": url if url != "local-corpus" else None, "contribution": score / total_score}
        for url, score in totals.items()
    ]
