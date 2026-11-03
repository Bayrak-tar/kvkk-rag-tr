"""Embedding sağlayıcısını JSON config ile kurar."""

from __future__ import annotations

import json
from pathlib import Path

from src.embeddings.sentence_transformer import SentenceTransformerEmbedder


def load_embedder(path: Path) -> SentenceTransformerEmbedder:
    """Model ve inference ayarlarını JSON dosyasından okuyup embedder oluşturur."""
    config = json.loads(path.read_text(encoding="utf-8"))
    return SentenceTransformerEmbedder(
        model_name=config["model_name"],
        device=config.get("device", "cpu"),
        batch_size=int(config.get("batch_size", 16)),
        normalize_embeddings=bool(config.get("normalize_embeddings", True)),
    )
