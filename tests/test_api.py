"""HTTP API sözleşmesini yapay RAG servisiyle sınar."""

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.generation.rag import Citation, RAGAnswer
from src.ui.app import ask_api


class _FakeService:
    def __init__(self) -> None:
        self.questions = []

    def answer(self, question: str) -> RAGAnswer:
        self.questions.append(question)
        citation = Citation(
            label="S1",
            law_name="6698 sayılı Kanun",
            article_no="11",
            paragraph_no="1",
            source_url="https://example.test/law",
        )
        return RAGAnswer(answer="Haklar [S1].", citations=[citation], retrieved=[])


def test_health_endpoint_does_not_initialize_rag_service() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_returns_answer_with_structured_citation() -> None:
    service = _FakeService()
    client = TestClient(create_app(service=service))

    response = client.post("/ask", json={"question": "İlgili kişinin hakları nelerdir?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Haklar [S1]."
    assert response.json()["citations"][0]["article_no"] == "11"
    assert service.questions == ["İlgili kişinin hakları nelerdir?"]


def test_ask_endpoint_rejects_blank_or_oversized_questions() -> None:
    client = TestClient(create_app(service=_FakeService()))

    blank = client.post("/ask", json={"question": "   "})
    oversized = client.post("/ask", json={"question": "x" * 2001})

    assert blank.status_code == 422
    assert oversized.status_code == 422


def test_gradio_client_rejects_blank_question_without_network_call() -> None:
    assert ask_api("  ", api_url="http://unused.test/ask") == ("Lütfen bir soru yazın.", "")
