"""Config'ten ortak soru-cevap servisi kuran çalışma zamanı fabrikası."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.embeddings.config import load_embedder
from src.generation.providers import ChatProvider, load_chat_provider
from src.generation.rag import RAGAnswer, Reranker, answer_question
from src.retrieval.indexing import load_rag_config
from src.retrieval.search import Retriever, load_retriever
from src.retrieval.store import load_vector_store


@dataclass(slots=True)
class RAGService:
    """Config edilmiş retriever, generator ve cevap limitini bir araya getirir."""

    retriever: Retriever
    reranker: Reranker | None
    chat_provider: ChatProvider
    top_k: int
    candidate_k: int | None
    verify_citations: bool = False

    def answer(self, question: str) -> RAGAnswer:
        """Seçilen RAG pipeline'ını tek soru için çalıştırır."""
        return answer_question(
            question,
            retriever=self.retriever,
            reranker=self.reranker,
            chat_provider=self.chat_provider,
            top_k=self.top_k,
            candidate_k=self.candidate_k,
            verify_citations=self.verify_citations,
        )


def build_rag_service(config_path: Path) -> RAGService:
    """Yerel RAG config'inden model ve retrieval bileşenlerini kurar."""
    config = load_rag_config(config_path)
    retrieval_config = config.get("retrieval", {})
    strategy = retrieval_config.get("strategy", "vector")
    embedder = (
        load_embedder(config["embedding_config"])
        if strategy in {"vector", "hybrid"}
        else None
    )
    vector_store = (
        load_vector_store(config["vector_store_config"], model_name=embedder.model_name)
        if embedder is not None
        else None
    )
    retriever, reranker = load_retriever(
        retrieval_config,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index_path=config["bm25_index_path"],
    )
    return RAGService(
        retriever=retriever,
        reranker=reranker,
        chat_provider=load_chat_provider(config["generation_config"]),
        top_k=int(config.get("top_k", 5)),
        candidate_k=(
            int(retrieval_config.get("candidate_k", 20)) if reranker is not None else None
        ),
        verify_citations=bool(config.get("citation_verification", {}).get("enabled", False)),
    )
