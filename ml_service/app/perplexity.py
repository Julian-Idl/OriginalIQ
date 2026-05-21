from __future__ import annotations

import math
from threading import Lock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import get_settings

_tokenizer = None
_model = None
_lock = Lock()


def _load():
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        device = get_settings().device
        _tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        _model.eval()
    return _tokenizer, _model


def perplexity(text: str, max_length: int = 512) -> float:
    tokenizer, model = _load()
    device = get_settings().device
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return 0.0

    with _lock, torch.no_grad():
        output = model(input_ids, labels=input_ids)
        loss = output.loss
    value = float(torch.exp(loss).detach().cpu())
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value

