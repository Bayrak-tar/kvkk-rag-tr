"""Evaluation JSONL şemasını doğrulama ve yükleme yardımcıları."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    """Elle doğrulanmış bir retrieval ve cevap değerlendirme örneği."""

    question_id: str
    question: str
    gold_article_ids: list[str]
    reference_answer: str
    source_urls: list[str]
    category: str


def _parse_question(item: dict[str, Any], *, line_number: int) -> EvaluationQuestion:
    """Bir JSON nesnesini alanlarını kontrol ederek evaluation örneğine çevirir."""
    required = (
        "id",
        "question",
        "gold_article_ids",
        "reference_answer",
        "source_urls",
        "status",
    )
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"Satır {line_number}: zorunlu alanlar eksik: {', '.join(missing)}")
    for field in ("id", "question", "reference_answer"):
        if not isinstance(item[field], str) or not item[field].strip():
            raise ValueError(f"Satır {line_number}: {field} boş olmayan metin olmalı.")
    gold_ids = item["gold_article_ids"]
    source_urls = item["source_urls"]
    if not isinstance(gold_ids, list) or not gold_ids or not all(
        isinstance(value, str) and ":" in value for value in gold_ids
    ):
        raise ValueError(f"Satır {line_number}: gold_article_ids en az bir 'source:article' olmalı.")
    if not isinstance(source_urls, list) or not source_urls or not all(
        isinstance(value, str) and value.startswith(("https://", "http://"))
        for value in source_urls
    ):
        raise ValueError(f"Satır {line_number}: source_urls geçerli URL listesi olmalı.")
    if item["status"] not in {"draft", "validated", "excluded"}:
        raise ValueError(f"Satır {line_number}: status draft, validated veya excluded olmalı.")
    return EvaluationQuestion(
        question_id=item["id"].strip(),
        question=item["question"].strip(),
        gold_article_ids=list(dict.fromkeys(gold_ids)),
        reference_answer=item["reference_answer"].strip(),
        source_urls=source_urls,
        category=str(item.get("category", "")),
    )


def load_questions(
    path: Path,
    *,
    statuses: set[str] | None = None,
) -> list[EvaluationQuestion]:
    """JSONL soru bankasını yükler; varsayılan olarak yalnızca validated örnekleri alır."""
    accepted_statuses = statuses or {"validated"}
    if not accepted_statuses <= {"draft", "validated", "excluded"}:
        raise ValueError("Bilinmeyen evaluation kayıt durumu.")
    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Satır {line_number}: geçersiz JSON: {error.msg}") from error
        if not isinstance(item, dict):
            raise ValueError(f"Satır {line_number}: her kayıt JSON nesnesi olmalı.")
        parsed = _parse_question(item, line_number=line_number)
        if parsed.question_id in seen_ids:
            raise ValueError(f"Tekrarlanan soru kimliği: {parsed.question_id}")
        seen_ids.add(parsed.question_id)
        if item["status"] in accepted_statuses:
            questions.append(parsed)
    return questions
