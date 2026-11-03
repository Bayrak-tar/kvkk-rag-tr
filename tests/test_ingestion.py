"""Ingestion ve mevzuat parser'ı için offline birim testleri."""

from src.ingestion.cleaning import html_to_text, normalize_text
from src.ingestion.models import LegalUnit
from src.ingestion.parser import parse_legal_units


def test_html_to_text_removes_scripts_and_keeps_block_boundaries() -> None:
    text = html_to_text("<h1>Kanun</h1><script>gizli()</script><p>MADDE 1 – (1) Metin.</p>")

    assert "gizli" not in text
    assert "Kanun" in text
    assert "MADDE 1" in text


def test_normalize_text_preserves_paragraphs_and_collapses_spaces() -> None:
    assert normalize_text("  Bir   metin.\r\n\r\n\r\nİkinci\tmetin. ") == (
        "Bir metin.\n\nİkinci metin."
    )


def test_parser_creates_article_and_paragraph_metadata() -> None:
    text = (
        "Amaç\nMADDE 1 – (1) Birinci\nfıkra.\n(2) İkinci fıkra.\n\n"
        "İKİNCİ BÖLÜM\n\nKişisel Veriler\n\nKapsam\nMADDE 2 – Kısa madde."
    )
    units = parse_legal_units(
        text,
        source_id="test-law",
        law_name="Örnek Kanun",
        source_url="https://example.test/law",
    )

    assert [(unit.article_no, unit.paragraph_no) for unit in units] == [
        ("1", "1"),
        ("1", "2"),
        ("2", None),
    ]
    assert units[0].title == "Amaç"
    assert units[0].text == "Birinci fıkra."
    assert units[1].text == "İkinci fıkra."
    assert "Kişisel Veriler" not in units[1].text
    assert units[2].title == "Kapsam"
    assert units[0].source_url == "https://example.test/law"


def test_unit_json_round_trip_is_stable() -> None:
    unit = LegalUnit.create(
        source_id="test",
        law_name="Örnek Kanun",
        article_no="1",
        paragraph_no="1",
        title=None,
        text="Metin",
        source_url="https://example.test",
    )

    assert LegalUnit.from_json(unit.to_json()) == unit
    assert LegalUnit.create(
        source_id="test",
        law_name="Örnek Kanun",
        article_no="1",
        paragraph_no="1",
        title=None,
        text="Metin",
        source_url="https://example.test",
    ).unit_id == unit.unit_id
