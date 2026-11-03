"""Vektör deposu arayüzü ve kalıcı Chroma uygulaması."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class VectorHit:
    """Aramadan dönen chunk, metadata ve uzaklık."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class VectorStore(Protocol):
    """Embedding vektörlerini ekleme ve sorgulama arayüzü."""

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Kayıtları kimliklerine göre ekler veya günceller."""

    def query(self, embedding: list[float], *, top_k: int) -> list[VectorHit]:
        """Sorgu vektörüne en yakın kayıtları döndürür."""


def _to_chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma'nın primitive metadata alanlarına uygun serileştirme yapar."""
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = json.dumps(value, ensure_ascii=False)
    return normalized


def _from_chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """JSON olarak saklanan liste metadata'sını Python değerine geri çevirir."""
    restored = dict(metadata)
    for key, value in restored.items():
        if isinstance(value, str) and value[:1] in {"[", "{"}:
            try:
                restored[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    return restored


class ChromaVectorStore:
    """Yerel diskte kalıcı Chroma koleksiyonunu kullanır."""

    def __init__(self, path: Path, collection: str, *, distance: str = "cosine") -> None:
        if distance not in {"cosine", "l2", "ip"}:
            raise ValueError("distance cosine, l2 veya ip olmalı.")
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError(
                "Vektör indeksi için `pip install -e '.[retrieval]'` çalıştırın."
            ) from error
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": distance},
            embedding_function=None,
        )

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Kayıtları Chroma'ya yazar; tekrar çalıştırma aynı kimlikleri günceller."""
        if not ids:
            return
        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("ids, documents, metadatas ve embeddings uzunlukları eşleşmeli.")
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=[_to_chroma_metadata(item) for item in metadatas],
            embeddings=embeddings,
        )

    def query(self, embedding: list[float], *, top_k: int) -> list[VectorHit]:
        """Chroma sıralamasından sonuçları dönüştürür."""
        if top_k <= 0:
            raise ValueError("top_k pozitif olmalı.")
        collection_size = self._collection.count()
        if collection_size == 0:
            return []
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, collection_size),
            include=["documents", "metadatas", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            VectorHit(
                chunk_id=chunk_id,
                text=document,
                metadata=_from_chroma_metadata(metadata or {}),
                distance=float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    def delete_except(self, keep_ids: set[str]) -> int:
        """Remove stale chunk IDs after a successful index rebuild."""
        existing = self._collection.get(include=[]).get("ids", [])
        stale_ids = [chunk_id for chunk_id in existing if chunk_id not in keep_ids]
        if stale_ids:
            self._collection.delete(ids=stale_ids)
        return len(stale_ids)


def load_vector_store(
    config_path: Path, *, model_name: str | None = None
) -> ChromaVectorStore:
    """Chroma kalıcı deposunu JSON config dosyasına göre oluşturur."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("provider") != "chroma":
        raise ValueError(f"Desteklenmeyen vector store: {config.get('provider')}")
    path = Path(config["path"])
    if not path.is_absolute():
        path = config_path.resolve().parents[1] / path
    collection = config["collection"]
    template = config.get("collection_template")
    if template and model_name:
        model_slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        collection = template.format(model_slug=model_slug)
    return ChromaVectorStore(
        path=path,
        collection=collection,
        distance=config.get("distance", "cosine"),
    )
