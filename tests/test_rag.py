"""RAG'nin bağlam ve kaynak atıf davranışını sınar."""

from src.generation.rag import answer_question
from src.retrieval.store import VectorHit


class _Embedder:
    def embed_queries(self, texts):
        return [[0.1, 0.2] for _ in texts]


class _Store:
    def __init__(self, hits):
        self.hits = hits

    def query(self, embedding, *, top_k):
        assert embedding == [0.1, 0.2]
        return self.hits[:top_k]


class _Chat:
    def __init__(self, result="Cevap [S1]."):
        self.result = result
        self.calls = []

    def generate(self, *, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.result


def _hit() -> VectorHit:
    return VectorHit(
        chunk_id="chunk-1",
        text="İlgili kişi, verisinin işlenip işlenmediğini öğrenebilir.",
        metadata={
            "law_name": "6698 sayılı Kişisel Verilerin Korunması Kanunu",
            "article_no": "11",
            "paragraph_no": ["1", "2"],
            "source_url": "https://example.test/kvkk",
        },
        distance=0.1,
    )


def test_answer_question_uses_retrieved_context_and_returns_citation() -> None:
    chat = _Chat()
    result = answer_question(
        "İlgili kişinin hakları nelerdir?",
        embedder=_Embedder(),
        vector_store=_Store([_hit()]),
        chat_provider=chat,
    )

    assert "GETİRİLEN BAĞLAM" in chat.calls[0][1]
    assert "6698 sayılı" in result.answer
    assert result.citations[0].article_no == "11"
    assert result.citations[0].paragraph_no == "1,2"


def test_answer_question_returns_not_found_without_calling_llm() -> None:
    chat = _Chat()
    result = answer_question(
        "Boş indeks sorusu",
        embedder=_Embedder(),
        vector_store=_Store([]),
        chat_provider=chat,
    )

    assert result.answer == "İlgili bağlamda bulunamadı."
    assert result.citations == []
    assert chat.calls == []


def test_answer_question_uses_reranker_output_and_candidate_count() -> None:
    class _Retriever:
        requested_k = None

        def search(self, query, *, top_k):
            del query
            self.requested_k = top_k
            return [_hit(), VectorHit("other", "İkinci kaynak", _hit().metadata, 0.2)]

    class _Reranker:
        def rerank(self, query, hits, *, top_k):
            del query, top_k
            return list(reversed(hits))

    retriever = _Retriever()
    result = answer_question(
        "Haklar nelerdir?",
        retriever=retriever,
        reranker=_Reranker(),
        candidate_k=9,
        chat_provider=_Chat(),
        top_k=1,
    )

    assert retriever.requested_k == 9
    assert [hit.chunk_id for hit in result.retrieved] == ["other"]
