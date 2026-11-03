"""Kaynaklı RAG cevapları sunan FastAPI uygulaması."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Annotated, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generation.rag import RAGAnswer
from src.generation.runtime import build_rag_service


class AnswerService(Protocol):
    """FastAPI'nin kullandığı minimal soru-cevap servisi arayüzü."""

    def answer(self, question: str) -> RAGAnswer:
        """Soruyu cevaplar ve kaynaklarını döndürür."""


class AskRequest(BaseModel):
    """Soru endpoint'i isteği."""

    question: Annotated[str, Field(min_length=1, max_length=2000)]


class CitationResponse(BaseModel):
    """Bir mevzuat madde atfı."""

    label: str
    law_name: str
    article_no: str
    paragraph_no: str | None
    source_url: str
    excerpt: str
    unit_type: str = "article"


class AskResponse(BaseModel):
    """RAG cevabı ve kaynak listesi."""

    answer: str
    citations: list[CitationResponse]
    verification_status: str
    latency_ms: float


def create_app(
    *,
    service: AnswerService | None = None,
    config_path: Path | None = None,
) -> FastAPI:
    """Test veya uygulama için isteğe bağlı dependency injection ile FastAPI kurar."""
    app = FastAPI(
        title="kvkk-rag-tr API",
        description="Kamuya açık Türk mevzuatına dayalı, kaynak atıflı soru-cevap API'si.",
        version="0.1.0",
    )
    app.state.rag_service = service
    selected_config = config_path or Path(
        os.getenv("KVKK_RAG_CONFIG", "configs/rag.json")
    )

    def get_service() -> AnswerService:
        """İlk soru geldiğinde modeli bir kez kurar ve uygulama durumunda saklar."""
        if app.state.rag_service is None:
            try:
                app.state.rag_service = build_rag_service(selected_config)
            except Exception as error:
                raise HTTPException(
                    status_code=503,
                    detail=f"RAG servisi başlatılamadı: {error}",
                ) from error
        return app.state.rag_service

    @app.get("/health")
    def health() -> dict[str, str]:
        """Model yüklemeden proses sağlık durumunu bildirir."""
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        """Kısa olmayan soruyu RAG servisine iletip kaynaklarıyla döndürür."""
        if not request.question.strip():
            raise HTTPException(status_code=422, detail="Soru boş olamaz.")
        started_at = perf_counter()
        try:
            result = get_service().answer(request.question.strip())
        except HTTPException:
            raise
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        answer_text = result.answer
        normalized_answer = answer_text.strip().casefold().replace("i\u0307", "i")
        no_relevant_context = normalized_answer.startswith("ilgili bağlamda bulunamadı")
        if no_relevant_context:
            answer_text = answer_text.split("\n\nKaynaklar:", maxsplit=1)[0]
        citations = [
            CitationResponse(
                label=citation.label,
                law_name=citation.law_name,
                article_no=citation.article_no,
                paragraph_no=citation.paragraph_no,
                source_url=citation.source_url,
                excerpt=citation.excerpt,
                unit_type=citation.unit_type,
            )
            for citation in ([] if no_relevant_context else result.citations)
        ]
        return AskResponse(
            answer=answer_text,
            citations=citations,
            verification_status=(
                "abstained" if no_relevant_context else result.verification_status
            ),
            latency_ms=round((perf_counter() - started_at) * 1000, 1),
        )

    return app


app = create_app()
