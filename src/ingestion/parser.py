"""Temizlenmiş mevzuat metnini madde ve fıkra kayıtlarına ayırır."""

from __future__ import annotations

import re

from src.ingestion.cleaning import normalize_text
from src.ingestion.models import LegalUnit

_ARTICLE_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?P<temporary>GEÇİCİ\s+)?MADDE\s+"
    r"(?P<number>\d+)\s*(?:[-–—:]\s*)?"
)
_PARAGRAPH_RE = re.compile(r"(?:^|\s)\((\d+)\)\s*")


def _split_trailing_heading(text: str) -> tuple[str, str | None]:
    """Madde sonuna eklenmiş noktalanmamış bölüm başlıklarını ayırır."""
    stripped = text.rstrip()
    heading: str | None = None
    while match := re.search(r"\n\s*\n([^\n]{1,100})$", stripped):
        candidate = match.group(1).strip()
        if candidate.endswith((".", ";", ":", "!", "?")) or len(candidate.split()) > 10:
            break
        stripped = stripped[: match.start()].rstrip()
        if heading is None:
            heading = candidate
    return stripped, heading


def _collapse_soft_line_breaks(text: str) -> str:
    """HTML'de satır sonuna bölünmüş cümleleri tek boşluğa indirger."""
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def parse_legal_units(
    text: str,
    *,
    source_id: str,
    law_name: str,
    source_url: str,
) -> list[LegalUnit]:
    """Mevzuat metnini madde/fıkra birimlerine ayırır ve kaynak metadata'sı ekler."""
    normalized = normalize_text(text)
    article_matches = list(_ARTICLE_RE.finditer(normalized))
    units: list[LegalUnit] = []
    leading_heading: str | None = None
    if article_matches:
        leading_lines = [
            line.strip() for line in normalized[: article_matches[0].start()].splitlines()
        ]
        leading_lines = [line for line in leading_lines if line]
        if leading_lines and len(leading_lines[-1]) <= 100:
            leading_heading = leading_lines[-1]
    next_heading = leading_heading

    for index, match in enumerate(article_matches):
        end = (
            article_matches[index + 1].start()
            if index + 1 < len(article_matches)
            else len(normalized)
        )
        article_text, trailing_heading = _split_trailing_heading(normalized[match.end() : end])
        article_text = _collapse_soft_line_breaks(article_text.strip())
        article_no = (
            f"Geçici {match.group('number')}"
            if match.group("temporary")
            else match.group("number")
        )
        paragraph_matches = list(_PARAGRAPH_RE.finditer(article_text))

        if not paragraph_matches:
            if article_text:
                units.append(
                    LegalUnit.create(
                        source_id=source_id,
                        law_name=law_name,
                        article_no=article_no,
                        paragraph_no=None,
                        title=next_heading,
                        text=article_text,
                        source_url=source_url,
                    )
                )
            next_heading = trailing_heading
            continue

        heading = article_text[: paragraph_matches[0].start()].strip(" \n:–—-") or None
        if heading is None:
            heading = next_heading
        next_heading = trailing_heading
        for paragraph_index, paragraph_match in enumerate(paragraph_matches):
            paragraph_end = (
                paragraph_matches[paragraph_index + 1].start()
                if paragraph_index + 1 < len(paragraph_matches)
                else len(article_text)
            )
            paragraph_text = article_text[paragraph_match.end() : paragraph_end].strip()
            if paragraph_text:
                units.append(
                    LegalUnit.create(
                        source_id=source_id,
                        law_name=law_name,
                        article_no=article_no,
                        paragraph_no=paragraph_match.group(1),
                        title=heading,
                        text=paragraph_text,
                        source_url=source_url,
                    )
                )
    return units


def parse_guidance_units(
    text: str,
    *,
    source_id: str,
    law_name: str,
    source_url: str,
    max_chars: int = 1400,
) -> list[LegalUnit]:
    """Kurum rehberi sayfasını başlık/paragraf gruplarına ayırır.

    Sayfada MADDE yapısı yoksa kullanılır; numaralar bölüm referansıdır, kanun maddesi değildir.
    """
    normalized = normalize_text(text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    ignored = {
        "tr", "en", "de", "mevzuat", "kurumsal", "yayınlar", "etkinlikler",
        "duyurular", "ilgili kişi", "veri sorumlusu", "görüntüle", "ana sayfa",
    }
    sections: list[tuple[str, list[str]]] = []
    current_title = law_name
    current: list[str] = []
    for line in lines:
        folded = line.casefold()
        if line.startswith("##"):
            if current:
                sections.append((current_title, current))
            current_title = line.lstrip("# ").strip()
            current = []
            continue
        if folded in ignored or "kişisel verileri koruma kurumu" == folded:
            continue
        if len(line) <= 120 and not line.endswith((".", ";", ":", "!", "?")):
            if current:
                sections.append((current_title, current))
            current_title = line
            current = [line]
            continue
        current.append(line)
    if current:
        sections.append((current_title, current))

    units: list[LegalUnit] = []
    section_number = 0
    for title, paragraphs in sections:
        buffer = ""
        for paragraph in paragraphs:
            if buffer and len(buffer) + len(paragraph) + 1 > max_chars:
                section_number += 1
                units.append(LegalUnit.create(
                    source_id=source_id, law_name=law_name,
                    article_no=f"Bölüm {section_number}", paragraph_no=None,
                    title=title, text=buffer, source_url=source_url, unit_type="guidance",
                ))
                buffer = ""
            if len(paragraph) > max_chars:
                for start in range(0, len(paragraph), max_chars):
                    section_number += 1
                    units.append(LegalUnit.create(
                        source_id=source_id, law_name=law_name,
                        article_no=f"Bölüm {section_number}", paragraph_no=None,
                        title=title, text=paragraph[start:start + max_chars],
                        source_url=source_url, unit_type="guidance",
                    ))
            else:
                buffer = f"{buffer}\n{paragraph}".strip()
        if buffer:
            section_number += 1
            units.append(LegalUnit.create(
                source_id=source_id, law_name=law_name,
                article_no=f"Bölüm {section_number}", paragraph_no=None,
                title=title, text=buffer, source_url=source_url, unit_type="guidance",
            ))
            buffer = ""
    return units
