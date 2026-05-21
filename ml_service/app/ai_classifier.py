from __future__ import annotations

from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import get_settings

_tokenizer = None
_model = None
_lock = Lock()


def _resolve_model_name() -> str | None:
    configured = get_settings().ai_classifier_model
    if Path(configured).exists():
        return configured
    if "/" in configured and not configured.startswith("models/"):
        return configured
    return None


def _load():
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        model_name = _resolve_model_name()
        if model_name is None:
            return None, None
        device = get_settings().device
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        _model.eval()
    return _tokenizer, _model


def roberta_ai_probability(text: str, max_length: int = 512) -> float:
    tokenizer, model = _load()
    if tokenizer is None or model is None:
        return 0.5
    device = get_settings().device
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with _lock, torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
    return float(probs[1]) if len(probs) > 1 else float(probs[0])
