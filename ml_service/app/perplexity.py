from __future__ import annotations

import math
from threading import Lock

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import get_settings

_tokenizer = None
_model = None
_lock = Lock()


def _load():
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        settings = get_settings()
        device = settings.device
        _tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=settings.local_files_only)
        _model = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=settings.local_files_only).to(device)
        if _tokenizer.pad_token_id is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model.config.pad_token_id = _tokenizer.pad_token_id
        if device == "cuda" and settings.inference_fp16:
            _model = _model.half()
        _model.eval()
    return _tokenizer, _model


def perplexities(texts: list[str], max_length: int = 512, batch_size: int | None = None) -> list[float]:
    if not texts:
        return []
    tokenizer, model = _load()
    settings = get_settings()
    device = settings.device
    batch_size = batch_size or settings.ai_batch_size
    values: list[float] = []

    with _lock, torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            if input_ids.shape[1] < 2:
                values.extend(0.0 for _ in range(input_ids.shape[0]))
                continue

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda" and settings.inference_fp16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            shift_logits = logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:].float()
            token_losses = F.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                reduction="none",
            )
            sequence_losses = (token_losses * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp_min(1.0)
            batch_values = torch.exp(sequence_losses).detach().cpu().tolist()
            for value in batch_values:
                values.append(0.0 if math.isnan(value) or math.isinf(value) else float(value))
    return values


def perplexity(text: str, max_length: int = 512) -> float:
    return perplexities([text], max_length=max_length, batch_size=1)[0]
