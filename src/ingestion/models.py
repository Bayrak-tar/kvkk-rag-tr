"""Mevzuat kayıtları için küçük, doğrulanabilir veri modelleri."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class LegalUnit:
    """Bir mevzuat maddesinin fıkrası veya fıkra bulunmayan madde metni."""

    unit_id: str
    source_id: str
    law_name: str
    article_no: str
    paragraph_no: str | None
    title: str | None
    text: str
    source_url: str
    unit_type: str = "article"

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        law_name: str,
        article_no: str,
        paragraph_no: str | None,
        title: str | None,
        text: str,
        source_url: str,
        unit_type: str = "article",
    ) -> LegalUnit:
        """Alanlardan kararlı bir kayıt kimliği üretip mevzuat birimini oluşturur."""
        key = "|".join(
            (source_id, article_no, paragraph_no or "", title or "", text, source_url, unit_type)
        )
        unit_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        return cls(
            unit_id=unit_id,
            source_id=source_id,
            law_name=law_name,
            article_no=article_no,
            paragraph_no=paragraph_no,
            title=title,
            text=text,
            source_url=source_url,
            unit_type=unit_type,
        )

    def to_json(self) -> str:
        """Kaydı UTF-8 uyumlu JSON satırı olarak serileştirir."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> LegalUnit:
        """JSON satırından mevzuat birimi oluşturur."""
        data = json.loads(line)
        data.setdefault("unit_type", "article")
        return cls(**data)
