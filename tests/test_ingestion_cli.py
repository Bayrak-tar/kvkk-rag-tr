"""İndirme/işleme akışını ağ çağrısı yapmadan sınar."""

import json

from src.ingestion import cli
from src.ingestion.models import LegalUnit


def test_ingest_sources_saves_raw_and_processed_data(tmp_path, monkeypatch) -> None:
    html = b"<h1>Ornek</h1><p>MADDE 1 - (1) Aciklama.</p>"
    monkeypatch.setattr(cli, "download_source", lambda url: html)
    source = {
        "id": "sample",
        "name": "Ornek Kanun",
        "law_name": "Ornek Kanun",
        "url": "https://example.test/law",
        "publisher": "Ornek kurum",
        "format": "html",
    }
    raw_dir = tmp_path / "raw"
    processed_path = tmp_path / "processed" / "units.jsonl"

    units = cli.ingest_sources([source], raw_dir=raw_dir, processed_path=processed_path)

    assert len(units) == 1
    assert (raw_dir / "sample.html").read_bytes() == html
    saved = LegalUnit.from_json(processed_path.read_text(encoding="utf-8").strip())
    assert saved == units[0]
    assert json.loads(processed_path.read_text(encoding="utf-8"))["source_url"] == source["url"]


def test_ingest_sources_can_process_cached_html_offline(tmp_path, monkeypatch) -> None:
    html = b"<p>MADDE 1 - (1) Aciklama.</p>"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sample.html").write_bytes(html)
    monkeypatch.setattr(
        cli, "download_source", lambda url: (_ for _ in ()).throw(AssertionError(url))
    )
    source = {
        "id": "sample",
        "name": "Ornek Kanun",
        "law_name": "Ornek Kanun",
        "url": "https://example.test/law",
        "publisher": "Ornek kurum",
        "format": "html",
    }

    units = cli.ingest_sources(
        [source], raw_dir=raw_dir, processed_path=tmp_path / "units.jsonl", download=False
    )

    assert len(units) == 1
