"""Sentence Transformers tabanlı embedding adaptörü."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """Model adını ve inference ayarlarını kullanarak embedding üretir."""

    model_name: str
    device: str = "cpu"
    batch_size: int = 16
    normalize_embeddings: bool = True
    _model: Any = None

    def _load_model(self) -> Any:
        """SentenceTransformer modelini ilk kullanımda yükler."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Embedding modelleri için `pip install -e '.[retrieval]'` çalıştırın."
                ) from error
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts: list[str], *, role: str) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        prefix = "query: " if role == "query" else "passage: "
        inputs = [f"{prefix}{text}" for text in texts] if self._uses_e5_prefix() else texts
        vectors = model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _uses_e5_prefix(self) -> bool:
        """E5 model ailesinde query/passage talimat öneklerinin gerekip gerekmediğini döndürür."""
        return self.model_name.lower().startswith("intfloat/multilingual-e5")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Sorguları E5 için query önekiyle, diğer modellerde doğrudan kodlar."""
        return self._encode(texts, role="query")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Mevzuat parçalarını E5 için passage önekiyle kodlar."""
        return self._encode(texts, role="passage")
