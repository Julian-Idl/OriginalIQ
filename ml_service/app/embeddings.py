from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import get_settings

_model: SentenceTransformer | None = None
_cache: "EmbeddingCache | None" = None
_lock = Lock()


class EmbeddingCache:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
        )
        self.conn.commit()
        self.lock = Lock()

    def get(self, key: str) -> np.ndarray | None:
        with self.lock:
            row = self.conn.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return np.asarray(json.loads(row[0]), dtype=np.float32)

    def set(self, key: str, vector: np.ndarray) -> None:
        payload = json.dumps(vector.astype(float).tolist())
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                (key, payload),
            )
            self.conn.commit()


def _cache_key(model_name: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model_name}:{digest}"


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        settings = get_settings()
        _model = SentenceTransformer(
            settings.sentence_model,
            device=settings.device,
            local_files_only=settings.local_files_only,
        )
        if settings.device == "cuda" and settings.inference_fp16:
            _model = _model.half()
    return _model


def get_cache() -> EmbeddingCache:
    global _cache
    if _cache is None:
        _cache = EmbeddingCache(get_settings().embedding_cache_path)
    return _cache


def embed_texts(texts: list[str], batch_size: int | None = None, normalize: bool = True) -> np.ndarray:
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    settings = get_settings()
    batch_size = batch_size or settings.embedding_batch_size
    cache = get_cache()
    vectors: list[np.ndarray | None] = []
    misses: list[tuple[int, str, str]] = []

    for idx, text in enumerate(texts):
        key = _cache_key(settings.sentence_model, text)
        vector = cache.get(key)
        vectors.append(vector)
        if vector is None:
            misses.append((idx, key, text))

    if misses:
        with _lock:
            model = get_embedding_model()
            encoded = model.encode(
                [item[2] for item in misses],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            ).astype(np.float32)
        for (idx, key, _text), vector in zip(misses, encoded):
            cache.set(key, vector)
            vectors[idx] = vector

    return np.vstack([vector for vector in vectors if vector is not None]).astype(np.float32)
