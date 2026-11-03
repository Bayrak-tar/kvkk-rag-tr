"""Config değerlerinden chunker seçmek için fabrika fonksiyonu."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from src.chunking.strategies import ArticleChunker, Chunker, FixedSizeChunker, SemanticChunker


def build_chunker(
    strategy: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 150,
    max_chars: int = 4000,
    threshold: float = 0.35,
    similarity: Callable[[str, str], float] | None = None,
) -> Chunker:
    """Strateji adına göre chunker döndürür; semantic için benzerlik işlevi ister."""
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "article":
        return ArticleChunker(max_chars=max_chars)
    if strategy == "semantic":
        if similarity is None:
            raise ValueError("Semantic chunking için similarity fonksiyonu gereklidir.")
        return SemanticChunker(similarity=similarity, threshold=threshold, max_chars=max_chars)
    raise ValueError(f"Bilinmeyen chunking stratejisi: {strategy}")


def load_chunker_config(
    path: Path, *, similarity: Callable[[str, str], float] | None = None
) -> Chunker:
    """JSON config içindeki seçili stratejiyi oluşturur."""
    config = json.loads(path.read_text(encoding="utf-8"))
    strategy = config.get("strategy")
    settings = config.get(strategy, {})
    if not isinstance(settings, dict):
        raise ValueError(f"'{strategy}' chunking ayarları JSON nesnesi olmalı.")
    return build_chunker(strategy, **settings, similarity=similarity)
