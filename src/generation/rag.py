"""Sorgu retrieval, bağlam oluşturma ve kaynaklı cevap üretme akışı."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from src.embeddings.base import Embedder
from src.generation.providers import ChatProvider
from src.retrieval.search import Retriever, VectorRetriever
from src.retrieval.store import VectorHit, VectorStore

SYSTEM_PROMPT = """Sen Türk mevzuatı üzerinde çalışan bir soru-cevap sistemisin.
Yalnızca kullanıcı mesajındaki GETİRİLEN BAĞLAM içindeki bilgilere dayan.
Bağlam soruyu yanıtlamıyorsa yalnızca 'İlgili bağlamda bulunamadı.' yaz.
Her hukuki iddia cümlesinin sonunda, iddiayı destekleyen [S1] gibi kaynak etiketlerini belirt.
Bağlamda olmayan kanun, madde, yorum veya olgu uydurma; hukuki tavsiye verme."""

_CITATION_LABEL = re.compile(r"\[(S\d+)\]")
_VERIFICATION_SYSTEM_PROMPT = """Sen Türk mevzuatı yanıtları için katı bir kanıt denetçisisin.
Yalnızca verilen yanıtı ve kaynak metinlerini kullan; dış bilginle boşlukları doldurma.
Yanıttaki her maddi iddianın, o cümlede açıkça gösterilen en az bir kaynak metni tarafından
doğrudan desteklenip desteklenmediğini denetle. Kaynak etiketi geçerli olsa bile metin iddiayı
desteklemiyorsa desteklenmiyor say. Yalnızca JSON döndür: {"supported": true/false,
"unsupported_claims": ["..."], "unknown_labels": ["S99"]}."""


@dataclass(frozen=True, slots=True)
class Citation:
    """Yanıtta kullanılan mevzuat kaynağı."""

    label: str
    law_name: str
    article_no: str
    paragraph_no: str | None
    source_url: str
    excerpt: str = ""
    unit_type: str = "article"

    def format(self) -> str:
        """Atfı kanun ve madde/fıkra bilgisiyle okunabilir hale getirir."""
        paragraph = f", fıkra {self.paragraph_no}" if self.paragraph_no else ""
        reference = "bölüm" if self.unit_type == "guidance" else "madde"
        return (
            f"[{self.label}] {self.law_name}, {reference} {self.article_no}{paragraph} "
            f"— {self.source_url}"
        )


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """Üretilen cevap, kaynaklar ve retrieval sonuçları."""

    answer: str
    citations: list[Citation]
    retrieved: list[VectorHit]
    verification_status: str = "not_checked"


class Reranker(Protocol):
    """Retriever adaylarını yeniden sıralama arayüzü."""

    def rerank(self, query: str, hits: list[VectorHit], *, top_k: int) -> list[VectorHit]:
        """En alakalı adayları döndürür."""


def _citations(hits: list[VectorHit]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for hit in hits:
        metadata: dict[str, Any] = hit.metadata
        law_name = str(metadata.get("law_name", "Mevzuat kaynağı"))
        article_no = str(metadata.get("article_no", "?"))
        paragraph_value = metadata.get("paragraph_no")
        paragraph_no = (
            ",".join(str(value) for value in paragraph_value)
            if isinstance(paragraph_value, list) and paragraph_value
            else str(paragraph_value) if paragraph_value else None
        )
        source_url = str(metadata.get("source_url", ""))
        key = (law_name, article_no, paragraph_no, source_url)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                label=f"S{len(citations) + 1}",
                law_name=law_name,
                article_no=article_no,
                paragraph_no=paragraph_no,
                source_url=source_url,
                excerpt=(
                    hit.text[:360].rsplit(" ", maxsplit=1)[0].strip()
                    + (" …" if len(hit.text) > 360 else "")
                ),
                unit_type=str(metadata.get("unit_type", "article")),
            )
        )
    return citations


def _make_context(hits: list[VectorHit], citations: list[Citation]) -> str:
    citation_by_key = {
        (
            citation.law_name,
            citation.article_no,
            citation.paragraph_no,
            citation.source_url,
        ): citation
        for citation in citations
    }
    sections: list[str] = []
    for hit in hits:
        metadata = hit.metadata
        law_name = str(metadata.get("law_name", "Mevzuat kaynağı"))
        article_no = str(metadata.get("article_no", "?"))
        paragraph_value = metadata.get("paragraph_no")
        paragraph_no = (
            ",".join(str(value) for value in paragraph_value)
            if isinstance(paragraph_value, list) and paragraph_value
            else str(paragraph_value) if paragraph_value else None
        )
        source_url = str(metadata.get("source_url", ""))
        citation = citation_by_key[(law_name, article_no, paragraph_no, source_url)]
        sections.append(f"[{citation.label}] {citation.format()}\n{hit.text}")
    return "\n\n".join(sections)


def _used_citations(answer: str, citations: list[Citation]) -> tuple[list[Citation], bool]:
    """Yanıttaki kaynak etiketlerini bulur ve etiketi olmayan/uydurma atıfları yakalar."""
    mentioned = set(_CITATION_LABEL.findall(answer))
    available = {citation.label for citation in citations}
    if not mentioned or not mentioned <= available:
        return [], False
    return [citation for citation in citations if citation.label in mentioned], True


def _verify_citation_support(
    *,
    question: str,
    answer: str,
    citations: list[Citation],
    hits: list[VectorHit],
    provider: ChatProvider,
) -> bool:
    """LLM denetçisine her iddiayı sadece atıf yaptığı kaynak metinle kontrol ettirir."""
    citation_by_key = {
        (item.law_name, item.article_no, item.paragraph_no, item.source_url): item
        for item in citations
    }
    evidence: dict[str, list[str]] = {citation.label: [] for citation in citations}
    for hit in hits:
        metadata = hit.metadata
        law_name = str(metadata.get("law_name", "Mevzuat kaynağı"))
        article_no = str(metadata.get("article_no", "?"))
        paragraph_value = metadata.get("paragraph_no")
        paragraph_no = (
            ",".join(str(value) for value in paragraph_value)
            if isinstance(paragraph_value, list) and paragraph_value
            else str(paragraph_value) if paragraph_value else None
        )
        source_url = str(metadata.get("source_url", ""))
        citation = citation_by_key.get((law_name, article_no, paragraph_no, source_url))
        if citation is not None and citation.label in evidence:
            evidence[citation.label].append(hit.text)

    payload = {
        "question": question,
        "answer": answer,
        "cited_sources": [
            {
                "label": citation.label,
                "law_name": citation.law_name,
                "article_no": citation.article_no,
                "paragraph_no": citation.paragraph_no,
                "text": "\n".join(evidence[citation.label])[:1800],
            }
            for citation in citations
        ],
    }
    try:
        response = provider.generate(
            system_prompt=_VERIFICATION_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
        ).strip()
    except Exception:
        return False
    if response.startswith("```"):
        response = (
            response.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
    json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if json_match:
        response = json_match.group(0)
    try:
        decision = json.loads(response)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(decision, dict)
        and decision.get("supported") is True
        and decision.get("unsupported_claims") == []
        and decision.get("unknown_labels") == []
    )


def answer_question(
    question: str,
    *,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    chat_provider: ChatProvider,
    top_k: int = 5,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    candidate_k: int | None = None,
    verify_citations: bool = False,
) -> RAGAnswer:
    """Vektör araması yapar ve yalnızca getirilen kaynak bağlamıyla cevap üretir."""
    if not question.strip():
        raise ValueError("Soru boş olamaz.")
    if top_k <= 0:
        raise ValueError("top_k pozitif olmalı.")
    if retriever is None:
        if embedder is None or vector_store is None:
            raise ValueError("retriever veya embedder ile vector_store birlikte verilmelidir.")
        retriever = VectorRetriever(embedder, vector_store)
    retrieval_count = candidate_k or (max(top_k, top_k * 4) if reranker else top_k)
    hits = retriever.search(question.strip(), top_k=retrieval_count)
    if reranker is not None:
        hits = reranker.rerank(question.strip(), hits, top_k=top_k)[:top_k]
    else:
        hits = hits[:top_k]
    if not hits:
        return RAGAnswer(
            answer="İlgili bağlamda bulunamadı.",
            citations=[],
            retrieved=[],
            verification_status="abstained",
        )
    citations = _citations(hits)
    context = _make_context(hits, citations)
    user_prompt = f"GETİRİLEN BAĞLAM:\n{context}\n\nSORU:\n{question.strip()}"
    generated = chat_provider.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    normalized_answer = " ".join(generated.strip().casefold().split()).replace(
        "i\u0307", "i"
    )
    if normalized_answer.startswith("ilgili bağlamda bulunamadı"):
        return RAGAnswer(
            answer=generated,
            citations=[],
            retrieved=hits,
            verification_status="abstained",
        )
    used_citations, labels_valid = _used_citations(generated, citations)
    if not labels_valid:
        return RAGAnswer(
            answer="Getirilen bağlamdan geçerli kaynak atıflı bir yanıt üretilemedi.",
            citations=[],
            retrieved=hits,
            verification_status="failed",
        )
    if verify_citations and not _verify_citation_support(
        question=question.strip(),
        answer=generated,
        citations=used_citations,
        hits=hits,
        provider=chat_provider,
    ):
        return RAGAnswer(
            answer=(
                "Getirilen kaynaklar yanıtı yeterince destekleyemedi; "
                "bu nedenle kaynaklı yanıt sunulmuyor."
            ),
            citations=[],
            retrieved=hits,
            verification_status="failed",
        )
    source_list = "\n".join(citation.format() for citation in used_citations)
    answer = f"{generated}\n\nKaynaklar:\n{source_list}" if source_list else generated
    return RAGAnswer(
        answer=answer,
        citations=used_citations,
        retrieved=hits,
        verification_status="passed" if verify_citations else "not_checked",
    )
