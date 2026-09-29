from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from apt_classifier.config import Settings

DEFAULT_EMBEDDING_MODEL = "basel/ATTACK-BERT"


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    revision: str | None = None
    cache_dir: str | Path | None = None
    device: str | None = None
    batch_size: int = 64
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_folder: str | None = None
        if self.cache_dir is not None:
            cache = Path(self.cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            cache_folder = str(cache)
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=cache_folder,
            revision=self.revision,
            trust_remote_code=False,
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, *, device: str | None = None, batch_size: int = 64
    ) -> "SentenceTransformerEmbedder":
        return cls(
            model_name=settings.embedding_model,
            revision=settings.embedding_model_revision,
            cache_dir=settings.model_cache_dir,
            device=device,
            batch_size=batch_size,
        )

    @property
    def dimension(self) -> int:
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Embedding model did not report its output dimension")
        return int(dimension)

    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        embeddings = self._model.encode(
            values,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        array = np.divide(array, norms, out=np.zeros_like(array), where=norms != 0)
        return array.tolist()

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        return self.embed(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


@dataclass(frozen=True, slots=True)
class ChromaEmbeddingFunction:
    embedder: Embedder

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return self.embedder.embed(input)

    @staticmethod
    def name() -> str:
        return "sentence-transformer-explicit"

    def get_config(self) -> dict[str, Any]:
        return {}


ChromaEmbedderAdapter = ChromaEmbeddingFunction
