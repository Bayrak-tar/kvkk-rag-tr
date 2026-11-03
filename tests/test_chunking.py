"""Chunking stratejileri için deterministik birim testleri."""

import pytest

from src.chunking.config import build_chunker, load_chunker_config
from src.chunking.strategies import ArticleChunker, FixedSizeChunker, SemanticChunker
from src.ingestion.models import LegalUnit


def _unit(article: str, paragraph: str, text: str) -> LegalUnit:
    return LegalUnit.create(
        source_id="law",
        law_name="Örnek Kanun",
        article_no=article,
        paragraph_no=paragraph,
        title=None,
        text=text,
        source_url="https://example.test/law",
    )


def test_fixed_chunking_respects_size_and_overlap() -> None:
    chunks = FixedSizeChunker(chunk_size=6, overlap=2).chunk([_unit("1", "1", "abcdefghijkl")])

    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij", "ijkl"]
    assert all(chunk.metadata["article_no"] == "1" for chunk in chunks)


def test_article_chunking_groups_only_same_article() -> None:
    chunks = ArticleChunker().chunk(
        [_unit("1", "1", "İlk."), _unit("1", "2", "İkinci."), _unit("2", "1", "Diğer.")]
    )

    assert len(chunks) == 2
    assert chunks[0].text == "İlk.\nİkinci."
    assert chunks[0].metadata["paragraph_no"] == ["1", "2"]


def test_semantic_chunking_uses_injected_similarity_function() -> None:
    chunker = SemanticChunker(similarity=lambda left, right: 0.9, threshold=0.5)
    chunks = chunker.chunk([_unit("1", "1", "Benzer."), _unit("1", "2", "Metin.")])

    assert len(chunks) == 1
    assert chunks[0].text == "Benzer.\nMetin."


def test_semantic_chunking_splits_when_similarity_is_low() -> None:
    chunker = SemanticChunker(similarity=lambda left, right: 0.1, threshold=0.5)
    chunks = chunker.chunk([_unit("1", "1", "Bir konu."), _unit("1", "2", "Başka konu.")])

    assert len(chunks) == 2


def test_chunker_factory_requires_semantic_similarity_provider() -> None:
    with pytest.raises(ValueError, match="similarity"):
        build_chunker("semantic")


def test_chunker_can_be_selected_from_json_config(tmp_path) -> None:
    config_path = tmp_path / "chunking.json"
    config_path.write_text(
        '{"strategy":"fixed","fixed":{"chunk_size":8,"overlap":2}}', encoding="utf-8"
    )

    chunker = load_chunker_config(config_path)

    assert isinstance(chunker, FixedSizeChunker)
    assert chunker.chunk_size == 8
