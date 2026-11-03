"""Türkçe metinler için hafif, yerel BM25 indeksi ve araması."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.retrieval.store import VectorHit

_TOKEN_PATTERN = re.compile(r"[^\w]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Noktalama işaretlerini ayırır ve Türkçe büyük I dönüşümünü korur."""
    normalized = text.replace("İ", "i").replace("I", "ı").lower()
    return [token for token in _TOKEN_PATTERN.split(normalized) if token]


@dataclass(frozen=True, slots=True)
class BM25Document:
    """BM25 indeksinde saklanan metin ve kaynak metadata'sı."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


class BM25Index:
    """JSONL olarak saklanabilen Okapi BM25 indeksi."""

    def __init__(
        self,
        documents: list[BM25Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 k1 pozitif, b ise 0 ile 1 arasında olmalı.")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(document.text) for document in documents]
        self._frequencies = [Counter(tokens) for tokens in self._tokens]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / len(self._lengths) if documents else 0.0
        document_frequency: Counter[str] = Counter()
        for frequencies in self._frequencies:
            document_frequency.update(frequencies.keys())
        self._document_frequency = document_frequency

    @classmethod
    def load(cls, path: Path, *, k1: float = 1.5, b: float = 0.75) -> BM25Index:
        """Chunk JSONL dosyasını belleğe alıp indeksi kurar."""
        documents = [
            BM25Document(
                chunk_id=item["chunk_id"],
                text=item["text"],
                metadata=item.get("metadata", {}),
            )
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for item in [json.loads(line)]
        ]
        return cls(documents, k1=k1, b=b)

    def save(self, path: Path) -> None:
        """Chunk kayıtlarını UTF-8 JSONL olarak yazar."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for document in self.documents:
                file.write(
                    json.dumps(
                        {
                            "chunk_id": document.chunk_id,
                            "text": document.text,
                            "metadata": document.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def search(self, query: str, *, top_k: int) -> list[VectorHit]:
        """BM25 skoruna göre en ilgili chunk'ları döndürür."""
        if top_k <= 0:
            raise ValueError("top_k pozitif olmalı.")
        query_terms = set(tokenize(query))
        if not query_terms or not self.documents:
            return []
        count = len(self.documents)
        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self._frequencies):
            score = 0.0
            length = self._lengths[index]
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = self._document_frequency[term]
                inverse_frequency = math.log1p((count - df + 0.5) / (df + 0.5))
                length_norm = self.k1 * (
                    1 - self.b + self.b * length / self._average_length
                )
                score += inverse_frequency * frequency * (self.k1 + 1) / (
                    frequency + length_norm
                )
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], self.documents[item[1]].chunk_id))
        return [
            VectorHit(
                chunk_id=self.documents[index].chunk_id,
                text=self.documents[index].text,
                metadata=self.documents[index].metadata,
                distance=-score,
            )
            for score, index in scored[:top_k]
        ]
