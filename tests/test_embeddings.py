"""Embedding adaptörünün model görevi ön eklerini sınar."""

from src.embeddings.sentence_transformer import SentenceTransformerEmbedder


class _Vectors(list):
    def tolist(self):
        return list(self)


class _FakeModel:
    def __init__(self) -> None:
        self.inputs = []

    def encode(self, texts, **kwargs):
        del kwargs
        self.inputs = texts
        return _Vectors([[1.0, 0.0] for _ in texts])


def test_e5_embedder_uses_query_and_passage_prefixes() -> None:
    model = _FakeModel()
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-base", _model=model)

    assert embedder.embed_queries(["haklarım nelerdir?"]) == [[1.0, 0.0]]
    assert model.inputs == ["query: haklarım nelerdir?"]
    assert embedder.embed_passages(["Kanun metni."]) == [[1.0, 0.0]]
    assert model.inputs == ["passage: Kanun metni."]


def test_bge_embedder_does_not_add_e5_prefixes() -> None:
    model = _FakeModel()
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", _model=model)

    embedder.embed_queries(["KVKK"])

    assert model.inputs == ["KVKK"]
