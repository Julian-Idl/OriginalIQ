from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from threading import Lock

import faiss
import numpy as np

from .config import get_settings


@dataclass(slots=True)
class RetrievalCandidate:
    text: str
    score: float
    source_url: str | None = None
    title: str | None = None
    metadata: dict | None = None


class FaissStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_path = Path(self.settings.faiss_index_path)
        self.metadata_path = Path(self.settings.faiss_metadata_path)
        self.lock = Lock()
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.metadata_path.exists():
            self.metadata = [
                json.loads(line)
                for line in self.metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def _ensure_index(self, dimension: int) -> None:
        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)

    def add(self, vectors: np.ndarray, records: list[dict], persist: bool = True) -> None:
        if vectors.size == 0:
            return
        with self.lock:
            self._ensure_index(vectors.shape[1])
            self.index.add(vectors.astype(np.float32))
            self.metadata.extend(records)
            if persist:
                self.save()

    def search(self, vectors: np.ndarray, top_k: int = 10) -> list[list[RetrievalCandidate]]:
        if self.index is None or self.index.ntotal == 0 or vectors.size == 0:
            return [[] for _ in range(len(vectors))]

        scores, indices = self.index.search(vectors.astype(np.float32), top_k)
        all_results: list[list[RetrievalCandidate]] = []
        for score_row, index_row in zip(scores, indices):
            row_results: list[RetrievalCandidate] = []
            for score, idx in zip(score_row, index_row):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[idx]
                row_results.append(
                    RetrievalCandidate(
                        text=meta.get("text", ""),
                        score=float(score),
                        source_url=meta.get("source_url"),
                        title=meta.get("title"),
                        metadata=meta,
                    )
                )
            all_results.append(row_results)
        return all_results

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            for record in self.metadata:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")


_store: FaissStore | None = None


def get_store() -> FaissStore:
    global _store
    if _store is None:
        _store = FaissStore()
    return _store


def records_from_texts(texts: list[str], source_url: str | None = None, title: str | None = None) -> list[dict]:
    return [
        {
            "text": text,
            "source_url": source_url,
            "title": title,
            "kind": "web" if source_url else "local",
        }
        for text in texts
    ]

