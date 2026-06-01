from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml_service"))

import torch

from app.ai_classifier import _load as load_roberta
from app.config import get_settings
from app.cross_encoder import get_cross_encoder
from app.embeddings import get_embedding_model
from app.perplexity import _load as load_gpt2


def model_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "unknown"


def main() -> None:
    settings = get_settings()
    print({"device_info": settings.device_info})

    embedding_model = get_embedding_model()
    print({"sentence_transformer_device": str(embedding_model.device)})

    cross_encoder = get_cross_encoder()
    print({"cross_encoder_device": model_device(cross_encoder.model)})

    _, roberta = load_roberta()
    print({"roberta_device": model_device(roberta) if roberta is not None else None})

    _, gpt2 = load_gpt2()
    print({"gpt2_device": model_device(gpt2)})

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(
            {
                "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
                "cuda_memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
            }
        )


if __name__ == "__main__":
    main()
