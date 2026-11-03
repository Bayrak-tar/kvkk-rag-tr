"""Chunk ve embedding ayarlarıyla kalıcı vektör indeksi oluşturur."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.embeddings.config import load_embedder
from src.retrieval.indexing import index_corpus, load_rag_config
from src.retrieval.store import load_vector_store


def main() -> None:
    """Config ile corpus'u embed edip Chroma koleksiyonuna yazar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rag.json"))
    args = parser.parse_args()
    config = load_rag_config(args.config)
    embedder = load_embedder(config["embedding_config"])
    vector_store = load_vector_store(
        config["vector_store_config"], model_name=embedder.model_name
    )
    count = index_corpus(
        corpus_path=config["corpus_path"],
        chunking_config=config["chunking_config"],
        embedder=embedder,
        vector_store=vector_store,
        bm25_index_path=config.get("bm25_index_path"),
    )
    print(f"Indexed {count} chunks.")


if __name__ == "__main__":
    main()
