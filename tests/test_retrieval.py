"""Corpus indexing akışını vektör deposu ve model olmadan sınar."""

import json
import sys
from types import ModuleType

from src.ingestion.models import LegalUnit
from src.retrieval.indexing import index_corpus
from src.retrieval.store import (
    ChromaVectorStore,
    _from_chroma_metadata,
    _to_chroma_metadata,
    load_vector_store,
)


class _FakeEmbedder:
    def embed_passages(self, texts):
        return [[float(len(texts)), 1.0] for _ in texts]


class _FakeStore:
    def __init__(self) -> None:
        self.calls = []

    def upsert(self, **kwargs):
        self.calls.append(kwargs)

    def query(self, embedding, *, top_k):
        del embedding, top_k
        return []


def test_metadata_lists_round_trip_for_chroma() -> None:
    original = {"article_no": "4", "paragraph_no": ["1", "2"], "title": None}

    encoded = _to_chroma_metadata(original)

    assert _from_chroma_metadata(encoded) == {"article_no": "4", "paragraph_no": ["1", "2"]}


def test_index_corpus_chunks_embeds_and_upserts(tmp_path) -> None:
    unit = LegalUnit.create(
        source_id="law",
        law_name="Örnek Kanun",
        article_no="1",
        paragraph_no="1",
        title="Amaç",
        text="Örnek mevzuat metni.",
        source_url="https://example.test/law",
    )
    corpus_path = tmp_path / "units.jsonl"
    corpus_path.write_text(unit.to_json() + "\n", encoding="utf-8")
    chunk_config = tmp_path / "chunking.json"
    chunk_config.write_text('{"strategy":"article","article":{"max_chars":1000}}', encoding="utf-8")
    store = _FakeStore()
    bm25_path = tmp_path / "bm25" / "chunks.jsonl"

    count = index_corpus(
        corpus_path=corpus_path,
        chunking_config=chunk_config,
        embedder=_FakeEmbedder(),
        vector_store=store,
        batch_size=1,
        bm25_index_path=bm25_path,
    )

    assert count == 1
    assert store.calls[0]["ids"]
    assert store.calls[0]["embeddings"] == [[1.0, 1.0]]
    assert '"chunk_id"' in bm25_path.read_text(encoding="utf-8")


def test_chroma_store_round_trips_metadata_and_model_collections(tmp_path, monkeypatch) -> None:
    class _Collection:
        def __init__(self) -> None:
            self.saved = {}

        def upsert(self, **kwargs) -> None:
            self.saved.update(kwargs)

        def count(self) -> int:
            return 1

        def query(self, **kwargs):
            assert kwargs["n_results"] == 1
            return {
                "ids": [["chunk"]],
                "documents": [["law text"]],
                "metadatas": [self.saved["metadatas"]],
                "distances": [[0.2]],
            }

    class _Client:
        def __init__(self) -> None:
            self.collection = _Collection()
            self.collection_name = None

        def get_or_create_collection(self, *, name, metadata, embedding_function):
            del metadata, embedding_function
            self.collection_name = name
            return self.collection

    client = _Client()
    chromadb = ModuleType("chromadb")
    chromadb.PersistentClient = lambda path: client
    monkeypatch.setitem(sys.modules, "chromadb", chromadb)
    config_path = tmp_path / "configs" / "vector.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "provider": "chroma",
                "path": "data/chroma",
                "collection": "kvkk_legal_texts",
                "collection_template": "kvkk_{model_slug}",
                "distance": "cosine",
            }
        ),
        encoding="utf-8",
    )

    store = load_vector_store(config_path, model_name="intfloat/multilingual-e5-base")
    assert isinstance(store, ChromaVectorStore)
    assert client.collection_name == "kvkk_intfloat_multilingual_e5_base"
    store.upsert(
        ids=["chunk"],
        documents=["law text"],
        metadatas=[{"paragraph_no": ["1", "2"]}],
        embeddings=[[0.1, 0.2]],
    )
    hit = store.query([0.1, 0.2], top_k=1)[0]

    assert hit.metadata["paragraph_no"] == ["1", "2"]
    assert hit.distance == 0.2
