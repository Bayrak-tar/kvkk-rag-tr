"""Config tabanlı retrieval ve isteğe bağlı kaynak atfı değerlendirmesi."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from eval.dataset import EvaluationQuestion, load_questions
from eval.metrics.faithfulness import citation_support_check
from eval.metrics.retrieval import mean_metrics, retrieval_metrics
from src.embeddings.config import load_embedder
from src.generation.providers import load_chat_provider
from src.generation.rag import answer_question
from src.retrieval.indexing import load_rag_config
from src.retrieval.search import CrossEncoderReranker, Retriever, load_retriever
from src.retrieval.store import VectorHit, load_vector_store


def _resolve(root: Path, value: str) -> Path:
    """Config'taki mutlak veya repo göreli yolu çözer."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    """Dosya içeriği için tekrar üretilebilir SHA-256 özeti döndürür."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _article_keys(hits: list[VectorHit]) -> list[str]:
    """Sonuçları tekrar etmeyen source/article kimlikleriyle gösterir."""
    keys: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        source_id = str(hit.metadata.get("source_id", ""))
        article_value = hit.metadata.get("article_no", "")
        articles = (
            article_value
            if isinstance(article_value, list)
            else str(article_value).split(",")
        )
        for article in articles:
            key = f"{source_id}:{str(article).strip()}"
            if source_id and str(article).strip() and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _construct_search(
    rag_config: dict[str, Any], retrieval_config: dict[str, Any]
) -> tuple[Retriever, CrossEncoderReranker | None, str | None]:
    """Yalnız seçilen retrieval stratejisinin gerekli modellerini kurar."""
    strategy = retrieval_config.get("strategy", "vector")
    embedder = load_embedder(rag_config["embedding_config"]) if strategy != "bm25" else None
    store = (
        load_vector_store(rag_config["vector_store_config"], model_name=embedder.model_name)
        if embedder is not None
        else None
    )
    retriever, reranker = load_retriever(
        retrieval_config,
        embedder=embedder,
        vector_store=store,
        bm25_index_path=rag_config["bm25_index_path"],
    )
    return retriever, reranker, embedder.model_name if embedder is not None else None


def _search_hits(
    question: str,
    *,
    retriever: Retriever,
    reranker: CrossEncoderReranker | None,
    top_k: int,
    candidate_k: int,
) -> list[VectorHit]:
    """Adayları getirir ve açıksa cross-encoder sıralaması uygular."""
    count = candidate_k if reranker is not None else top_k
    hits = retriever.search(question, top_k=count)
    return reranker.rerank(question, hits, top_k=top_k)[:top_k] if reranker else hits[:top_k]


def _evaluate_question(
    item: EvaluationQuestion,
    *,
    retriever: Retriever,
    reranker: CrossEncoderReranker | None,
    max_k: int,
    candidate_k: int,
    chat_provider: Any | None,
    verify_citations: bool,
) -> tuple[dict[str, Any], list[VectorHit]]:
    """Bir soru için retrieval ve opsiyonel üretim kontrolünü çalıştırır."""
    if chat_provider is None:
        hits = _search_hits(
            item.question,
            retriever=retriever,
            reranker=reranker,
            top_k=max_k,
            candidate_k=candidate_k,
        )
        answer = None
        faithfulness = None
        latency_ms = None
    else:
        started_at = perf_counter()
        rag_answer = answer_question(
            item.question,
            retriever=retriever,
            reranker=reranker,
            chat_provider=chat_provider,
            top_k=max_k,
            candidate_k=candidate_k if reranker else None,
            verify_citations=verify_citations,
        )
        hits = rag_answer.retrieved
        answer = rag_answer.answer
        faithfulness = citation_support_check(
            answer, {citation.label for citation in rag_answer.citations}
        )
        verification_status = rag_answer.verification_status
        latency_ms = round((perf_counter() - started_at) * 1000, 1)
    row: dict[str, Any] = {
        "question_id": item.question_id,
        "question": item.question,
        "reference_answer": item.reference_answer,
        "source_urls": item.source_urls,
        "category": item.category,
        "gold_article_ids": item.gold_article_ids,
        "retrieved_article_ids": _article_keys(hits),
        "answer": answer,
        "latency_ms": latency_ms,
        "citations": (
            [
                {
                    "label": citation.label,
                    "law_name": citation.law_name,
                    "article_no": citation.article_no,
                    "paragraph_no": citation.paragraph_no,
                    "source_url": citation.source_url,
                    "excerpt": citation.excerpt,
                }
                for citation in rag_answer.citations
            ]
            if chat_provider is not None
            else []
        ),
        "verification_status": (
            verification_status if chat_provider is not None else "not_run"
        ),
    }
    if faithfulness is not None:
        row.update(
            {
                "citation_coverage": faithfulness["citation_coverage"],
                "uncited_sentences": faithfulness["uncited_sentences"],
                "abstained": faithfulness["abstained"],
                "citation_claim_sentence_count": faithfulness["claim_sentence_count"],
            }
        )
    return row, hits


def run_experiment(config_path: Path) -> tuple[Path, Path]:
    """Eval config'ini çalıştırır ve ayrıntı CSV'si ile özet JSON üretir."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parents[1]
    rag_path = _resolve(root, config["rag_config"])
    rag_config = load_rag_config(rag_path)
    retrieval_config = dict(rag_config.get("retrieval", {}))
    retrieval_overrides = config.get("retrieval_overrides", {})
    retrieval_config.update(retrieval_overrides)
    if "reranker" in retrieval_overrides:
        retrieval_config["reranker"] = {
            **rag_config.get("retrieval", {}).get("reranker", {}),
            **retrieval_overrides["reranker"],
        }

    question_path = _resolve(root, config["questions_path"])
    selected_statuses = set(config.get("statuses", ["validated"]))
    questions = load_questions(question_path, statuses=selected_statuses)
    if not questions:
        raise ValueError(
            f"{question_path} içinde seçilen durumlarda soru yok. "
            "Taslakları uzman incelemesinden geçirip status alanını validated yapın."
        )

    ks = sorted({int(value) for value in config.get("ks", [1, 3, 5])})
    if not ks or ks[0] <= 0:
        raise ValueError("ks pozitif tam sayılardan oluşmalı.")
    max_k = max(ks)
    retriever, reranker, embedding_model = _construct_search(rag_config, retrieval_config)
    generation_enabled = bool(config.get("run_generation", False))
    chat_provider = (
        load_chat_provider(rag_config["generation_config"])
        if generation_enabled
        else None
    )
    verify_citations = bool(rag_config.get("citation_verification", {}).get("enabled", False))
    candidate_k = max(max_k, int(retrieval_config.get("candidate_k", max_k * 4)))

    rows: list[dict[str, Any]] = []
    for item in questions:
        row, hits = _evaluate_question(
            item,
            retriever=retriever,
            reranker=reranker,
            max_k=max_k,
            candidate_k=candidate_k,
            chat_provider=chat_provider,
            verify_citations=verify_citations,
        )
        for k in ks:
            row.update(retrieval_metrics(hits, item.gold_article_ids, k=k))
        rows.append(row)

    metric_names = [
        name
        for k in ks
        for name in (f"recall@{k}", f"mrr@{k}", f"ndcg@{k}")
    ]
    summary = mean_metrics([{name: row[name] for name in metric_names} for row in rows])
    if generation_enabled:
        scored_answers = [
            row["citation_coverage"]
            for row in rows
            if row["citation_coverage"] is not None
        ]
        summary["citation_coverage"] = (
            sum(scored_answers) / len(scored_answers) if scored_answers else None
        )
        summary["citation_coverage_answer_count"] = len(scored_answers)
        passed_count = sum(row["verification_status"] == "passed" for row in rows)
        summary["citation_verification_pass_rate"] = passed_count / len(rows)
        summary["citation_verification_passed_count"] = passed_count
        latencies = sorted(
            float(row["latency_ms"]) for row in rows if row["latency_ms"] is not None
        )
        summary["mean_latency_ms"] = round(mean(latencies), 1) if latencies else None
        summary["p95_latency_ms"] = (
            latencies[max(0, int(0.95 * len(latencies) + 0.999) - 1)]
            if latencies
            else None
        )

    output_dir = _resolve(root, config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = str(config["experiment_id"])
    csv_path = output_dir / f"{experiment_id}.csv"
    json_path = output_dir / f"{experiment_id}.json"
    columns = [
        "question_id",
        "question",
        "reference_answer",
        "source_urls",
        "category",
        "gold_article_ids",
        "retrieved_article_ids",
        *metric_names,
        "citation_coverage",
        "abstained",
        "citation_claim_sentence_count",
        "uncited_sentences",
        "citations",
        "answer",
        "latency_ms",
        "verification_status",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "gold_article_ids": json.dumps(row["gold_article_ids"], ensure_ascii=False),
                    "retrieved_article_ids": json.dumps(
                        row["retrieved_article_ids"], ensure_ascii=False
                    ),
                    "source_urls": json.dumps(row["source_urls"], ensure_ascii=False),
                    "citations": json.dumps(row.get("citations", []), ensure_ascii=False),
                    "uncited_sentences": json.dumps(
                        row.get("uncited_sentences", []), ensure_ascii=False
                    ),
                }
            )

    chunk_config = json.loads(rag_config["chunking_config"].read_text(encoding="utf-8"))
    metadata = {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "question_count": len(questions),
        "question_statuses": sorted(selected_statuses),
        "evaluation_validity": (
            "reviewed" if selected_statuses == {"validated"} else "preliminary_unreviewed"
        ),
        "question_set_sha256": _sha256(question_path),
        "rag_config_sha256": _sha256(rag_path),
        "corpus_sha256": _sha256(rag_config["corpus_path"]),
        "retrieval_strategy": retrieval_config.get("strategy", "vector"),
        "embedding_model": embedding_model,
        "chunking_strategy": chunk_config.get("strategy"),
        "reranker": retrieval_config.get("reranker", {}),
        "run_generation": generation_enabled,
        "citation_verification": verify_citations,
        "ks": ks,
    }
    result = {"metadata": metadata, "aggregate_metrics": summary, "per_question": rows}
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    """Command-line entry point for a single configured experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    csv_path, json_path = run_experiment(args.config)
    print("CSV:", csv_path, "JSON:", json_path)


if __name__ == "__main__":
    main()
