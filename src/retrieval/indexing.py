"""Chunk embedding'lerini vektör deposuna yazan corpus indexing akışı."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.chunking.config import load_chunker_config
from src.embeddings.base import Embedder
from src.ingestion.models import LegalUnit
from src.retrieval.bm25 import BM25Document, BM25Index
from src.retrieval.store import VectorStore


def load_legal_units(path: Path) -> list[LegalUnit]:
    """JSONL dosyasındaki mevzuat birimlerini yükler."""
    return [
        LegalUnit.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def index_corpus(
    *,
    corpus_path: Path,
    chunking_config: Path,
    embedder: Embedder,
    vector_store: VectorStore,
    batch_size: int = 64,
    bm25_index_path: Path | None = None,
) -> int:
    """JSONL mevzuat verisini chunk'lar, embed eder ve vector store'a yazar."""
    if batch_size <= 0:
        raise ValueError("batch_size pozitif olmalı.")
    units = load_legal_units(corpus_path)

    def similarity(left: str, right: str) -> float:
        vectors = embedder.embed_passages([left, right])
        if len(vectors) != 2:
            raise ValueError("Semantic chunking iki benzerlik vektörü bekliyor.")
        left_norm = math.sqrt(sum(value * value for value in vectors[0]))
        right_norm = math.sqrt(sum(value * value for value in vectors[1]))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        dot = sum(a * b for a, b in zip(vectors[0], vectors[1], strict=True))
        return dot / (left_norm * right_norm)

    chunker = load_chunker_config(chunking_config, similarity=similarity)
    chunks = chunker.chunk(units)
    if bm25_index_path is not None:
        BM25Index(
            [
                BM25Document(chunk.chunk_id, chunk.text, chunk.metadata)
                for chunk in chunks
            ]
        ).save(bm25_index_path)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = embedder.embed_passages([chunk.text for chunk in batch])
        if len(embeddings) != len(batch):
            raise ValueError("Embedder çıktı sayısı chunk sayısıyla eşleşmiyor.")
        vector_store.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch],
            embeddings=embeddings,
        )
    delete_stale = getattr(vector_store, "delete_except", None)
    if delete_stale is not None:
        delete_stale({chunk.chunk_id for chunk in chunks})
    return len(chunks)


def load_rag_config(path: Path) -> dict[str, Any]:
    """RAG ana config'ini okuyup göreli yolları repo köküne göre çözer."""
    if not path.is_absolute() and not path.is_file():
        project_path = Path(__file__).resolve().parents[2] / path
        if project_path.is_file():
            path = project_path
    config = json.loads(path.read_text(encoding="utf-8"))
    root = path.resolve().parents[1]
    for key in (
        "embedding_config",
        "vector_store_config",
        "generation_config",
        "chunking_config",
        "corpus_path",
    ):
        value = Path(config[key])
        config[key] = value if value.is_absolute() else root / value
    bm25_path = Path(config.get("bm25_index_path", "data/processed/bm25_chunks.jsonl"))
    config["bm25_index_path"] = bm25_path if bm25_path.is_absolute() else root / bm25_path
    config.setdefault("retrieval", {"strategy": "vector"})
    return config
