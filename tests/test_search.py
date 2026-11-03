"""BM25 ve hibrit arama davranışlarını küçük sabit corpus üzerinde sınar."""

from src.retrieval.bm25 import BM25Document, BM25Index, tokenize
from src.retrieval.search import HybridRetriever
from src.retrieval.store import VectorHit


def _hit(chunk_id: str) -> VectorHit:
    return VectorHit(
        chunk_id=chunk_id,
        text=f"Metin {chunk_id}",
        metadata={"article_no": chunk_id},
        distance=0.1,
    )


def test_tokenize_preserves_turkish_i_rules() -> None:
    assert tokenize("İŞLENEN VERİLER ve ISPARTA") == ["işlenen", "veriler", "ve", "ısparta"]


def test_bm25_finds_matching_document_and_round_trips(tmp_path) -> None:
    index = BM25Index(
        [
            BM25Document("a", "İlgili kişinin veriye erişim hakkı vardır", {"article_no": "11"}),
            BM25Document("b", "Veri sorumlusu güvenlik tedbirleri alır", {"article_no": "12"}),
        ]
    )
    path = tmp_path / "chunks.jsonl"

    index.save(path)
    loaded = BM25Index.load(path)
    results = loaded.search("erişim hakkı", top_k=2)

    assert results[0].chunk_id == "a"
    assert results[0].metadata["article_no"] == "11"
    assert results[0].distance < 0


class _StaticRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query: str, *, top_k: int) -> list[VectorHit]:
        del query
        return self.hits[:top_k]


def test_hybrid_rrf_favors_document_ranked_by_both_retrievers() -> None:
    hybrid = HybridRetriever(
        _StaticRetriever([_hit("shared"), _hit("vector-only")]),
        _StaticRetriever([_hit("shared"), _hit("bm25-only")]),
        rrf_k=60,
        candidate_k=3,
    )

    results = hybrid.search("soru", top_k=3)

    assert [hit.chunk_id for hit in results] == ["shared", "bm25-only", "vector-only"]
    assert results[0].distance < results[1].distance
