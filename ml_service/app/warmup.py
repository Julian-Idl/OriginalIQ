from __future__ import annotations

import time

import torch

from .ai_classifier import roberta_ai_probability
from .config import get_settings
from .cross_encoder import rerank
from .embeddings import embed_texts, get_embedding_model
from .ensemble import get_ensemble
from .perplexity import perplexity
from .preprocessing import get_nlp
from .retrieval import RetrievalCandidate, get_store

_status = {
    "ready": False,
    "started_at": None,
    "finished_at": None,
    "duration_seconds": None,
    "device": None,
    "steps": [],
    "error": None,
}


def warmup_status() -> dict:
    return dict(_status)


def _step(name: str, fn) -> None:
    started = time.perf_counter()
    fn()
    _status["steps"].append(
        {
            "name": name,
            "seconds": round(time.perf_counter() - started, 3),
        }
    )


def warmup_models() -> dict:
    if _status["ready"]:
        return warmup_status()

    started = time.perf_counter()
    _status.update(
        {
            "ready": False,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "finished_at": None,
            "duration_seconds": None,
            "device": get_settings().device,
            "steps": [],
            "error": None,
        }
    )

    try:
        sample = "OriginalIQ warms up plagiarism detection and AI writing analysis models."

        _step("spaCy NLP pipeline", lambda: get_nlp()(sample))
        _step("FAISS index and metadata", lambda: get_store())
        _step("Sentence-BERT embeddings", lambda: (get_embedding_model(), embed_texts([sample], batch_size=1)))
        _step(
            "cross-encoder reranker",
            lambda: rerank(
                sample,
                [RetrievalCandidate(text="OriginalIQ analyzes source similarity with reranking.", score=0.0)],
                top_n=1,
            ),
        )
        _step("fine-tuned RoBERTa classifier", lambda: roberta_ai_probability(sample))
        _step("GPT-2 perplexity model", lambda: perplexity(sample))
        _step("ensemble model", lambda: get_ensemble())

        if torch.cuda.is_available():
            _step("CUDA synchronization", lambda: torch.cuda.synchronize())

        _status["ready"] = True
        _status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _status["duration_seconds"] = round(time.perf_counter() - started, 3)
        return warmup_status()
    except Exception as error:
        _status["error"] = str(error)
        _status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _status["duration_seconds"] = round(time.perf_counter() - started, 3)
        raise

