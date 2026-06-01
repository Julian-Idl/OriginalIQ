from __future__ import annotations

from pathlib import Path
from threading import Lock

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

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
        settings = get_settings()
        device = settings.device
        _tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=settings.local_files_only)

        model_path = Path(model_name)
        if (model_path / "adapter_config.json").exists() and (model_path / "model.safetensors").exists():
            try:
                from peft import PeftModel
            except ImportError as error:
                raise ImportError("Loading the LoRA-trained AI detector requires peft.") from error

            config = AutoConfig.from_pretrained(model_name, local_files_only=settings.local_files_only)
            base_model = AutoModelForSequenceClassification.from_config(config)
            state_dict = load_file(str(model_path / "model.safetensors"))
            base_model.load_state_dict(state_dict, strict=False)
            _model = PeftModel.from_pretrained(
                base_model,
                model_name,
                local_files_only=settings.local_files_only,
            ).to(device)
        else:
            _model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=settings.local_files_only,
            ).to(device)
        if device == "cuda" and settings.inference_fp16:
            _model = _model.half()
        _model.eval()
    return _tokenizer, _model


def roberta_ai_probabilities(texts: list[str], max_length: int = 512, batch_size: int | None = None) -> list[float]:
    if not texts:
        return []
    tokenizer, model = _load()
    if tokenizer is None or model is None:
        return [0.5 for _ in texts]
    settings = get_settings()
    device = settings.device
    batch_size = batch_size or settings.ai_batch_size
    probabilities: list[float] = []

    with _lock, torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda" and settings.inference_fp16):
                logits = model(**encoded).logits
            probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
            probabilities.extend(float(row[1]) if len(row) > 1 else float(row[0]) for row in probs)
    return probabilities


def roberta_ai_probability(text: str, max_length: int = 512) -> float:
    return roberta_ai_probabilities([text], max_length=max_length, batch_size=1)[0]
