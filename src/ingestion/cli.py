"""Mevzuat kaynaklarını indirip işlenmiş JSONL kayıtları üretir."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.ingestion.cleaning import decode_html, html_to_text
from src.ingestion.models import LegalUnit
from src.ingestion.parser import parse_guidance_units, parse_legal_units

ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "kvkk-rag-tr/0.1 (public legal text research project)"


def load_sources(path: Path) -> list[dict[str, str]]:
    """JSON kaynak manifestini yükler ve temel alanlarını doğrular."""
    sources = json.loads(path.read_text(encoding="utf-8"))
    required = {"id", "name", "law_name", "url", "publisher", "format"}
    if not isinstance(sources, list) or not all(
        isinstance(source, dict) and required <= source.keys() for source in sources
    ):
        raise ValueError(f"Kaynak manifesti gerekli alanları içermeli: {sorted(required)}")
    return sources


def download_source(url: str, *, timeout: float = 30.0) -> bytes:
    """Resmî kaynağı belirli kullanıcı aracısı ve zaman aşımıyla indirir."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise URLError(f"HTTP durum kodu: {response.status}")
        return response.read()


def ingest_sources(
    sources: list[dict[str, str]],
    *,
    raw_dir: Path,
    processed_path: Path,
    download: bool = True,
) -> list[LegalUnit]:
    """HTML kaynaklarını indirir veya ham dosyalardan işleyerek JSONL üretir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    units: list[LegalUnit] = []

    for source in sources:
        if source["format"].lower() != "html":
            raise ValueError(f"Desteklenmeyen kaynak biçimi: {source['format']}")
        raw_path = raw_dir / f"{source['id']}.html"
        if download:
            payload = download_source(source["url"])
            raw_path.write_bytes(payload)
        else:
            if not raw_path.is_file():
                raise FileNotFoundError(f"Çevrimdışı ham kaynak bulunamadı: {raw_path}")
            payload = raw_path.read_bytes()
        html = decode_html(payload)
        visible_text = html_to_text(
            html,
            content_class=(
                "news__detail-article"
                if source.get("document_type") == "guidance"
                else None
            ),
        )
        if source.get("document_type") == "guidance":
            source_units = parse_guidance_units(
                visible_text,
                source_id=source["id"],
                law_name=source["law_name"],
                source_url=source["url"],
            )
        else:
            source_units = parse_legal_units(
                visible_text,
                source_id=source["id"],
                law_name=source["law_name"],
                source_url=source["url"],
            )
        if not source_units:
            raise ValueError(
                f"{source['id']} kaynağında madde bulunamadı; sayfa biçimini kontrol edin."
            )
        units.extend(source_units)
        print(f"{source['id']}: {len(source_units)} units")

    processed_path.write_text("\n".join(unit.to_json() for unit in units) + "\n", encoding="utf-8")
    return units


def main() -> None:
    """Komut satırından mevzuat indirme ve yapılandırma akışını çalıştırır."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "sources.json")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--processed", type=Path, default=ROOT / "data" / "processed" / "legal_units.jsonl"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Önceden indirilmiş data/raw dosyalarını kullan."
    )
    args = parser.parse_args()
    ingest_sources(
        load_sources(args.manifest),
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        download=not args.offline,
    )


if __name__ == "__main__":
    main()
