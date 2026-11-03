"""Kaynak atfı kapsama kontrolü; metin anlamını doğrulamaz."""

from __future__ import annotations

import re
from typing import Any

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION = re.compile(r"\[(S\d+)\]")
_NOT_FOUND = "bulunamadı"


def citation_support_check(answer: str, valid_labels: set[str]) -> dict[str, Any]:
    """İddia cümlelerinde geçerli kaynak etiketi olup olmadığını kabaca ölçer."""
    answer_body = answer.split("Kaynaklar:", maxsplit=1)[0]
    if _NOT_FOUND in answer_body.casefold():
        return {
            "citation_coverage": None,
            "claim_sentence_count": 0,
            "supported_sentence_count": 0,
            "uncited_sentences": [],
            "abstained": True,
            "check_type": "citation_coverage_heuristic",
        }
    claims: list[str] = []
    supported = 0
    uncited: list[str] = []
    for sentence in _SENTENCE_END.split(answer_body.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        claims.append(sentence)
        labels = set(_CITATION.findall(sentence))
        if labels & valid_labels:
            supported += 1
        else:
            uncited.append(sentence)
    score = supported / len(claims) if claims else 1.0
    return {
        "citation_coverage": score,
        "claim_sentence_count": len(claims),
        "supported_sentence_count": supported,
        "uncited_sentences": uncited,
        "abstained": False,
        "check_type": "citation_coverage_heuristic",
    }
