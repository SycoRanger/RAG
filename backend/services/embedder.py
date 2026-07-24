"""
backend/services/embedder.py
------------------------------
Wraps a local Sentence-Transformers model (PyTorch backend) so the rest of
the app never imports torch or sentence_transformers directly. The model
is loaded once and cached, since loading it is the expensive part.

Torch is a direct, explicit dependency here (see requirements.txt) — it's
what actually runs the embedding model's forward pass; sentence-transformers
is a convenience wrapper on top of it. torchvision is NOT used anywhere in
this project: there is no image/vision model in this pipeline (see the
README for why, and what would need to change to add OCR).
"""

from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from backend.config import settings


@lru_cache(maxsize=1)
def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name, device=_get_device())


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings. Returns one vector per input string."""
    if not texts:
        return []
    model = _get_model()
    with torch.no_grad():
        vectors = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # unit-norm vectors -> cosine == dot product
            convert_to_numpy=True,
        )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]


def embedding_dimension() -> int:
    return _get_model().get_sentence_embedding_dimension()


def device_in_use() -> str:
    return _get_device()
