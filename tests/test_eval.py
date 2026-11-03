"""Evaluation dataset, retrieval metrics, citation heuristic and reports tests."""

import json

import pytest

from eval.compare import compare
from eval.dataset import load_questions
from eval.metrics.faithfulness import citation_support_check
from eval.metrics.retrieval import retrieval_metrics
from src.retrieval.store import VectorHit


def _hit(chunk_id: str, article_no: str) -> VectorHit:
    return VectorHit(
        chunk_id=chunk_id,
        text=chunk_id,
        metadata={"source_id": "kvkk-6698", "article_no": article_no},
        distance=0.1,
    )


def test_question_loader_defaults_to_validated_only(tmp_path) -> None:
    path = tmp_path / "questions.jsonl"
    records = [
        {
            "id": "Q-1",
            "question": "Soru?",
            "gold_article_ids": ["kvkk-6698:11"],
            "reference_answer": "Yanıt.",
            "source_urls": ["https://example.test/law"],
            "status": "draft",
        },
        {
            "id": "Q-2",
            "question": "Soru 2?",
            "gold_article_ids": ["kvkk-6698:12"],
            "reference_answer": "Yanıt 2.",
            "source_urls": ["https://example.test/law"],
            "status": "validated",
        },
    ]
    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")

    assert [question.question_id for question in load_questions(path)] == ["Q-2"]
    assert len(load_questions(path, statuses={"draft", "validated"})) == 2


def test_retrieval_metrics_deduplicates_chunks_by_article() -> None:
    metrics = retrieval_metrics(
        [_hit("chunk-a", "10"), _hit("chunk-b", "10"), _hit("chunk-c", "11")],
        ["kvkk-6698:11"],
        k=2,
    )

    assert metrics == {"recall@2": 1.0, "mrr@2": 0.5, "ndcg@2": pytest.approx(0.6309, abs=1e-4)}


def test_citation_support_check_reports_uncited_claims() -> None:
    result = citation_support_check("İlk iddia [S1]. İkinci iddia.", {"S1"})

    assert result["citation_coverage"] == 0.5
    assert result["uncited_sentences"] == ["İkinci iddia."]
    assert result["check_type"] == "citation_coverage_heuristic"


def test_abstention_is_excluded_from_citation_coverage() -> None:
    result = citation_support_check("İlgili bağlamda bulunamadı.", set())

    assert result["citation_coverage"] is None
    assert result["abstained"] is True


def test_comparison_writes_only_from_existing_result_json(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "one.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "experiment_id": "bm25-one",
                    "retrieval_strategy": "bm25",
                    "question_count": 1,
                    "ks": [1],
                },
                "aggregate_metrics": {"recall@1": 0.5, "mrr@1": 0.5, "ndcg@1": 0.5},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "comparison"

    paths = compare(results, output)

    assert all(path.is_file() for path in paths)
    assert "bm25-one" in paths[1].read_text(encoding="utf-8")
    assert "0.500" in paths[2].read_text(encoding="utf-8")


def test_seed_dataset_has_40_draft_questions_and_default_load_is_empty() -> None:
    path = __import__("pathlib").Path("eval/questions/kvkk_seed.jsonl")

    assert len(load_questions(path, statuses={"draft"})) == 40
    assert load_questions(path) == []
