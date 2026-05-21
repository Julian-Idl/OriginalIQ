from functools import lru_cache
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    serpapi_api_key: str = ""
    sentence_model: str = "sentence-transformers/all-mpnet-base-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ai_classifier_model: str = "models/roberta-ai-detector"
    ensemble_model_path: str = "models/ensemble.joblib"
    faiss_index_path: str = "storage/faiss.index"
    faiss_metadata_path: str = "storage/faiss_metadata.jsonl"
    embedding_cache_path: str = "storage/embedding_cache.sqlite"
    spacy_model: str = "en_core_web_sm"
    max_web_results: int = 5
    top_k: int = 10
    web_trigger_threshold: float = 0.75

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _resolve_project_path(self, value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str(self.project_root / path)

    def model_post_init(self, __context) -> None:
        self.ai_classifier_model = self._resolve_project_path(self.ai_classifier_model)
        self.ensemble_model_path = self._resolve_project_path(self.ensemble_model_path)
        self.faiss_index_path = self._resolve_project_path(self.faiss_index_path)
        self.faiss_metadata_path = self._resolve_project_path(self.faiss_metadata_path)
        self.embedding_cache_path = self._resolve_project_path(self.embedding_cache_path)

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_dirs(self) -> None:
        for file_path in [
            self.faiss_index_path,
            self.faiss_metadata_path,
            self.embedding_cache_path,
            self.ensemble_model_path,
        ]:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
