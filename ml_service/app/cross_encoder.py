from __future__ import annotations

from threading import Lock

from sentence_transformers import CrossEncoder

from .config import get_settings

_model: CrossEncoder | None = None
_lock = Lock()


def get_cross_encoder() -> CrossEncoder:
    global _model
    if _model is None:
        settings = get_settings()
        _model = CrossEncoder(settings.cross_encoder_model, device=settings.device)
    return _model


def rerank(query: str, candidates: list, top_n: int = 5) -> list:
    if not candidates:
        return []

    pairs = [(query, candidate.text) for candidate in candidates]
    with _lock:
        scores = get_cross_encoder().predict(pairs, show_progress_bar=False)

    scored = []
    for candidate, score in zip(candidates, scores):
        candidate.score = float(score)
        scored.append(candidate)
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_n]

