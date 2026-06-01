from functools import lru_cache
import os
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_ROOT / ".env"), ".env"),
        extra="ignore",
        case_sensitive=False,
    )

    serpapi_api_key: str = ""
    sentence_model: str = "sentence-transformers/all-mpnet-base-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ai_classifier_model: str = "models/roberta-ai-detector"
    ensemble_model_path: str = "models/ensemble.joblib"
    faiss_index_path: str = "storage/faiss.index"
    faiss_metadata_path: str = "storage/faiss_metadata.jsonl"
    embedding_cache_path: str = "storage/embedding_cache.sqlite"
    spacy_model: str = "en_core_web_sm"
    hf_home: str = "models/huggingface"
    hf_hub_offline: bool = False
    transformers_offline: bool = False
    ml_device: str = "auto"
    inference_fp16: bool = True
    ai_batch_size: int = 8
    embedding_batch_size: int = 32
    cross_encoder_batch_size: int = 16
    max_web_results: int = 3
    max_web_search_chunks: int = 6
    web_search_concurrency: int = 3
    web_timeout_seconds: float = 12.0
    top_k: int = 10
    web_trigger_threshold: float = 0.75

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def _resolve_project_path(self, value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str(self.project_root / path)

    def model_post_init(self, __context) -> None:
        if self.hf_home:
            hf_home_path = self._resolve_project_path(self.hf_home)
            os.environ.setdefault("HF_HOME", hf_home_path)
        if self.hf_hub_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        if self.transformers_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.ai_classifier_model = self._resolve_project_path(self.ai_classifier_model)
        self.ensemble_model_path = self._resolve_project_path(self.ensemble_model_path)
        self.faiss_index_path = self._resolve_project_path(self.faiss_index_path)
        self.faiss_metadata_path = self._resolve_project_path(self.faiss_metadata_path)
        self.embedding_cache_path = self._resolve_project_path(self.embedding_cache_path)

    @property
    def device(self) -> str:
        requested = self.ml_device.strip().lower()
        if requested in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise RuntimeError("ML_DEVICE=cuda was requested, but PyTorch cannot see CUDA.")
            return "cuda"
        if requested == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device_info(self) -> dict:
        cuda_available = torch.cuda.is_available()
        return {
            "requested": self.ml_device,
            "selected": self.device,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "hf_home": os.environ.get("HF_HOME"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        }

    @property
    def local_files_only(self) -> bool:
        return self.hf_hub_offline or self.transformers_offline

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
