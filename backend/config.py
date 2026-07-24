"""
backend/config.py
------------------
Centralized configuration for the API backend. Every environment-dependent
value (API keys, model names, paths, defaults) lives here so the rest of
the codebase never touches os.environ directly.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent  # project root, one level above backend/


def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and not value:
        raise EnvironmentError(
            f"Missing required environment variable '{key}'. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


@dataclass(frozen=True)
class Settings:
    # --- Pinecone ---
    pinecone_api_key: str = field(default_factory=lambda: _get_env("PINECONE_API_KEY", required=True))
    pinecone_cloud: str = field(default_factory=lambda: _get_env("PINECONE_CLOUD", "aws"))
    pinecone_region: str = field(default_factory=lambda: _get_env("PINECONE_REGION", "us-east-1"))
    pinecone_index_name: str = field(default_factory=lambda: _get_env("PINECONE_INDEX_NAME", "rag-pdf-index"))
    pinecone_metric: str = "cosine"

    # --- Groq (LLM) ---
    groq_api_key: str = field(default_factory=lambda: _get_env("GROQ_API_KEY", required=True))
    # gpt-oss-20b is Groq's recommended fast/cheap general-purpose model as of mid-2026
    # (llama-3.1-8b-instant / llama-3.3-70b-versatile were deprecated). Override via .env.
    groq_model: str = field(default_factory=lambda: _get_env("GROQ_MODEL", "openai/gpt-oss-20b"))
    groq_temperature: float = 0.0  # deterministic, minimizes hallucination

    # --- Embeddings ---
    embedding_model_name: str = field(
        default_factory=lambda: _get_env("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    )
    embedding_dimension: int = 384

    # --- Chunking defaults (overridable per-request) ---
    default_chunk_size: int = 800
    default_chunk_overlap: int = 120

    # --- Retrieval defaults ---
    default_top_k: int = 5
    # 0.35 was too strict in practice: small embedding models like MiniLM
    # don't bridge simple rewordings well (e.g. a question asking "how many
    # days" against a document phrased in "weeks" can score well below 0.35
    # even though it's the right passage). 0.25 catches those while still
    # filtering genuinely irrelevant chunks. Override via .env if needed.
    default_similarity_threshold: float = field(
        default_factory=lambda: float(_get_env("SIMILARITY_THRESHOLD", "0.25"))
    )

    # --- OCR (for scanned / image-heavy PDFs) ---
    # "off" | "auto" | "force" — auto only OCRs pages with no real text layer.
    default_ocr_mode: str = field(default_factory=lambda: _get_env("OCR_MODE", "auto"))
    ocr_dpi: int = field(default_factory=lambda: int(_get_env("OCR_DPI", "300")))
    ocr_language: str = field(default_factory=lambda: _get_env("OCR_LANGUAGE", "eng"))
    # Windows: the Tesseract installer often skips PATH, so allow an explicit
    # path e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
    tesseract_cmd: str = field(default_factory=lambda: _get_env("TESSERACT_CMD", ""))

    # --- Upload limits ---
    max_upload_mb: int = 20

    # --- Paths (all relative to project root, not backend/) ---
    data_dir: Path = BASE_DIR / "data"
    log_dir: Path = BASE_DIR / "data" / "logs"
    query_log_path: Path = BASE_DIR / "data" / "logs" / "query_log.csv"
    registry_path: Path = BASE_DIR / "data" / "registry.json"

    # --- API server ---
    api_host: str = field(default_factory=lambda: _get_env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(_get_env("API_PORT", "8000")))
    cors_origins: list[str] = field(default_factory=lambda: _get_env("CORS_ORIGINS", "*").split(","))


settings = Settings()
settings.log_dir.mkdir(parents=True, exist_ok=True)
