"""Sabit boyut, madde ve benzerlik tabanlı chunking stratejileri."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from src.ingestion.models import LegalUnit


@dataclass(frozen=True, slots=True)
class Chunk:
    """Metin parçası ve üretildiği mevzuat birimlerinin metadata'sı."""

    chunk_id: str
    text: str
    metadata: dict[str, str | list[str] | None]


class Chunker(Protocol):
    """Mevzuat birimlerini chunk listesine dönüştüren arayüz."""

    def chunk(self, units: list[LegalUnit]) -> list[Chunk]:
        """Birimleri seçili stratejiye göre parçalara dönüştürür."""


def _metadata(unit: LegalUnit) -> dict[str, str | list[str] | None]:
    return {
        "source_id": unit.source_id,
        "law_name": unit.law_name,
        "article_no": unit.article_no,
        "paragraph_no": unit.paragraph_no,
        "title": unit.title,
        "source_url": unit.source_url,
        "unit_type": unit.unit_type,
    }


def _make_chunk(text: str, units: list[LegalUnit], suffix: str) -> Chunk:
    unit_ids = [unit.unit_id for unit in units]
    articles = list(dict.fromkeys(unit.article_no for unit in units))
    paragraphs = list(dict.fromkeys(unit.paragraph_no for unit in units if unit.paragraph_no))
    first = units[0]
    metadata = _metadata(first)
    metadata["article_no"] = articles[0] if len(articles) == 1 else ",".join(articles)
    metadata["paragraph_no"] = paragraphs
    metadata["unit_ids"] = unit_ids
    return Chunk(chunk_id=f"{unit_ids[0]}-{suffix}", text=text, metadata=metadata)


@dataclass(frozen=True, slots=True)
class FixedSizeChunker:
    """Metni karakter pencerelerine ayıran ve isteğe bağlı örtüşme ekleyen strateji."""

    chunk_size: int = 1200
    overlap: int = 150

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("chunk_size pozitif, overlap ise 0 <= overlap < chunk_size olmalı.")

    def chunk(self, units: list[LegalUnit]) -> list[Chunk]:
        """Her mevzuat birimini karakter uzunluğuna göre bölerek döndürür."""
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for unit in units:
            for index, start in enumerate(range(0, len(unit.text), step)):
                text = unit.text[start : start + self.chunk_size].strip()
                if text:
                    chunks.append(_make_chunk(text, [unit], str(index)))
        return chunks


@dataclass(frozen=True, slots=True)
class ArticleChunker:
    """Aynı kanun maddesinin fıkralarını başlık ve madde sınırlarını koruyarak birleştirir."""

    max_chars: int = 5000

    def chunk(self, units: list[LegalUnit]) -> list[Chunk]:
        """Madde/fıkra birimlerini, sınır ve boyut uygunsa, madde bazında birleştirir."""
        groups: list[list[LegalUnit]] = []
        for unit in units:
            if (
                groups
                and groups[-1][0].source_id == unit.source_id
                and groups[-1][0].article_no == unit.article_no
                and sum(len(item.text) for item in groups[-1]) + len(unit.text) <= self.max_chars
            ):
                groups[-1].append(unit)
            else:
                groups.append([unit])
        return [
            _make_chunk("\n".join(unit.text for unit in group), group, str(index))
            for index, group in enumerate(groups)
        ]


@dataclass(frozen=True, slots=True)
class SemanticChunker:
    """Komşu fıkraları benzerlik fonksiyonuna göre birleştiren semantic strategy."""

    similarity: Callable[[str, str], float]
    threshold: float = 0.35
    max_chars: int = 4000

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold 0 ile 1 arasında olmalı.")
        if self.max_chars <= 0:
            raise ValueError("max_chars pozitif olmalı.")

    def chunk(self, units: list[LegalUnit]) -> list[Chunk]:
        """Benzer ardışık fıkraları bir araya getirir; farklı maddeler arasında bölünür."""
        if not units:
            return []
        groups: list[list[LegalUnit]] = [[units[0]]]
        for unit in units[1:]:
            current = groups[-1]
            same_article = (
                current[0].source_id == unit.source_id
                and current[0].article_no == unit.article_no
            )
            current_text = "\n".join(item.text for item in current)
            within_limit = len(current_text) + len(unit.text) <= self.max_chars
            is_similar = self.similarity(current_text, unit.text) >= self.threshold
            if same_article and within_limit and is_similar:
                current.append(unit)
            else:
                groups.append([unit])
        return [
            _make_chunk("\n".join(unit.text for unit in group), group, f"semantic-{index}")
            for index, group in enumerate(groups)
        ]
