"""İndekslenmiş mevzuat üzerinde kaynaklı RAG sorusu çalıştırır."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.embeddings.config import load_embedder
from src.generation.providers import load_chat_provider
from src.generation.rag import answer_question
from src.retrieval.indexing import load_rag_config
from src.retrieval.search import load_retriever
from src.retrieval.store import load_vector_store


def main() -> None:
    """Config ile retrieval ve cevap üretimini başlatır."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="KVKK veya ilgili mevzuata ilişkin soru")
    parser.add_argument("--config", type=Path, default=Path("configs/rag.json"))
    args = parser.parse_args()
    config = load_rag_config(args.config)
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
    result = answer_question(
        args.question,
        retriever=retriever,
        reranker=reranker,
        chat_provider=load_chat_provider(config["generation_config"]),
        top_k=int(config.get("top_k", 5)),
        candidate_k=(
            int(retrieval_config.get("candidate_k", 20)) if reranker is not None else None
        ),
    )
    print(result.answer)


if __name__ == "__main__":
    main()
