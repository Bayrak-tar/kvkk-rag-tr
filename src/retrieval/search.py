"""Vektör, BM25 ve RRF hibrit arama stratejileri."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from src.embeddings.base import Embedder
from src.retrieval.bm25 import BM25Index
from src.retrieval.store import VectorHit, VectorStore


class Retriever(Protocol):
    """Metin sorgusunu ilgili mevzuat chunk'larına dönüştürür."""

    def search(self, query: str, *, top_k: int) -> list[VectorHit]:
        """Sıralanmış sonuçları döndürür."""


class VectorRetriever:
    """Embedding ve vektör deposuyla anlamsal arama yapar."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def search(self, query: str, *, top_k: int) -> list[VectorHit]:
        vectors = self.embedder.embed_queries([query])
        if len(vectors) != 1:
            raise ValueError("Embedder tek sorgu için tam bir vektör döndürmeli.")
        return self.vector_store.query(vectors[0], top_k=top_k)


class BM25Retriever:
    """İn-memory BM25 indeksini Retriever arayüzüne bağlar."""

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def search(self, query: str, *, top_k: int) -> list[VectorHit]:
        return self.index.search(query, top_k=top_k)


class HybridRetriever:
    """Vektör ve BM25 sıralamalarını Reciprocal Rank Fusion ile birleştirir."""

    def __init__(
        self,
        vector: Retriever,
        lexical: Retriever,
        *,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        if rrf_k <= 0 or candidate_k <= 0:
            raise ValueError("rrf_k ve candidate_k pozitif olmalı.")
        self.vector = vector
        self.lexical = lexical
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(self, query: str, *, top_k: int) -> list[VectorHit]:
        """İki arama listesinin rank skorlarını toplar ve tekrarları birleştirir."""
        if top_k <= 0:
            raise ValueError("top_k pozitif olmalı.")
        rankings = (
            self.vector.search(query, top_k=self.candidate_k),
            self.lexical.search(query, top_k=self.candidate_k),
        )
        hits: dict[str, VectorHit] = {}
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, hit in enumerate(ranking, start=1):
                hits.setdefault(hit.chunk_id, hit)
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (
                    self.rrf_k + rank
                )
        ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        return [
            VectorHit(
                chunk_id=chunk_id,
                text=hits[chunk_id].text,
                metadata=hits[chunk_id].metadata,
                distance=-scores[chunk_id],
            )
            for chunk_id in ordered[:top_k]
        ]


class CrossEncoderReranker:
    """CrossEncoder ile aday chunk'ları sorguya göre yeniden sıralar."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "CrossEncoder için `pip install -e '.[retrieval]'` çalıştırın."
            ) from error
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[VectorHit], *, top_k: int) -> list[VectorHit]:
        """Adayları sorgu-belge relevance skoruna göre azalan sıralar."""
        if top_k <= 0:
            raise ValueError("top_k pozitif olmalı.")
        if not hits:
            return []
        scores = self._model.predict([(query, hit.text) for hit in hits])
        ordered = sorted(
            zip(hits, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].chunk_id),
        )
        return [hit for hit, _score in ordered[:top_k]]


def load_retriever(
    config: dict[str, Any],
    *,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    bm25_index_path: Path | None = None,
) -> tuple[Retriever, CrossEncoderReranker | None]:
    """RAG retrieval ayarlarından retriever ve opsiyonel reranker kurar."""
    strategy = config.get("strategy", "vector")
    vector: VectorRetriever | None = None
    if strategy in {"vector", "hybrid"}:
        if embedder is None or vector_store is None:
            raise ValueError("Vector veya hybrid arama için embedder ve vector_store gerekir.")
        vector = VectorRetriever(embedder, vector_store)
    lexical = None
    if strategy in {"bm25", "hybrid"}:
        if bm25_index_path is None:
            raise ValueError("BM25 araması için bm25_index_path gerekir.")
        lexical = BM25Retriever(
            BM25Index.load(
                bm25_index_path,
                k1=float(config.get("bm25_k1", 1.5)),
                b=float(config.get("bm25_b", 0.75)),
            )
        )
    if strategy == "vector":
        assert vector is not None
        retriever: Retriever = vector
    elif strategy == "bm25":
        assert lexical is not None
        retriever = lexical
    elif strategy == "hybrid":
        assert lexical is not None
        assert vector is not None
        retriever = HybridRetriever(
            vector,
            lexical,
            rrf_k=int(config.get("rrf_k", 60)),
            candidate_k=int(config.get("candidate_k", 20)),
        )
    else:
        raise ValueError(f"Desteklenmeyen retrieval stratejisi: {strategy}")

    rerank_config = config.get("reranker", {})
    reranker = None
    if rerank_config.get("enabled", False):
        reranker = CrossEncoderReranker(
            str(rerank_config.get("model_name", "BAAI/bge-reranker-v2-m3"))
        )
    return retriever, reranker
