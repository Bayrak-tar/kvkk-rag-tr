"""Run a draft in-domain/OOD abstention experiment against the configured RAG service."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from src.generation.runtime import build_rag_service
from src.retrieval.indexing import load_rag_config


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cases(path: Path, statuses: set[str]) -> list[dict[str, Any]]:
    """Load and validate JSONL test cases; drafts remain explicitly labeled."""
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number}: expected a JSON object.")
        required = {"id", "question", "expected_abstain", "status"}
        if not required <= item.keys():
            raise ValueError(f"Line {line_number}: missing {sorted(required - item.keys())}.")
        if not isinstance(item["expected_abstain"], bool):
            raise ValueError(f"Line {line_number}: expected_abstain must be boolean.")
        if item["status"] not in {"draft", "validated", "excluded"}:
            raise ValueError(f"Line {line_number}: invalid status.")
        case_id = str(item["id"])
        if case_id in seen:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen.add(case_id)
        if item["status"] in statuses:
            cases.append(item)
    if not cases:
        raise ValueError("No abstention cases match the selected statuses.")
    if not any(case["expected_abstain"] for case in cases):
        raise ValueError("The selected set must include out-of-domain abstention cases.")
    if not any(not case["expected_abstain"] for case in cases):
        raise ValueError("The selected set must include answerable in-domain cases.")
    return cases


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Compute binary abstention metrics; positive means abstain is expected."""
    tp = sum(row["expected_abstain"] and row["predicted_abstain"] for row in rows)
    fp = sum(not row["expected_abstain"] and row["predicted_abstain"] for row in rows)
    fn = sum(row["expected_abstain"] and not row["predicted_abstain"] for row in rows)
    tn = sum(not row["expected_abstain"] and not row["predicted_abstain"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {
        "case_count": len(rows),
        "true_abstain": tp,
        "false_refusal": fp,
        "missed_abstention": fn,
        "answered_in_domain": tn,
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": f1,
        "accuracy": (tp + tn) / len(rows),
        "ood_unsafe_answer_rate": fn / (tp + fn) if tp + fn else None,
        "in_domain_false_refusal_rate": fp / (fp + tn) if fp + tn else None,
    }


def run_abstention_experiment(config_path: Path) -> tuple[Path, Path]:
    """Run one answerability experiment and save case-level and aggregate results."""
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parents[1]
    rag_path = Path(config["rag_config"])
    rag_path = rag_path if rag_path.is_absolute() else root / rag_path
    question_path = Path(config["questions_path"])
    question_path = question_path if question_path.is_absolute() else root / question_path
    statuses = set(config.get("statuses", ["validated"]))
    cases = _load_cases(question_path, statuses)

    rag_config = load_rag_config(rag_path)
    service = build_rag_service(rag_path)
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = perf_counter()
        result = service.answer(str(case["question"]))
        latency_ms = round((perf_counter() - started) * 1000, 1)
        predicted_abstain = result.verification_status in {"abstained", "failed"}
        rows.append(
            {
                "case_id": case["id"],
                "question": case["question"],
                "category": case.get("category", ""),
                "status": case["status"],
                "expected_abstain": case["expected_abstain"],
                "predicted_abstain": predicted_abstain,
                "correct": case["expected_abstain"] == predicted_abstain,
                "verification_status": result.verification_status,
                "citations": [
                    {
                        "law_name": citation.law_name,
                        "unit_type": citation.unit_type,
                        "article_no": citation.article_no,
                        "source_url": citation.source_url,
                    }
                    for citation in result.citations
                ],
                "answer": result.answer,
                "latency_ms": latency_ms,
            }
        )

    output_dir = Path(config.get("output_dir", "results/abstention"))
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = str(config["experiment_id"])
    csv_path = output_dir / f"{experiment_id}.csv"
    json_path = output_dir / f"{experiment_id}.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {**row, "citations": json.dumps(row["citations"], ensure_ascii=False)}
            )
    result = {
        "metadata": {
            "experiment_id": experiment_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "evaluation_validity": (
                "reviewed" if statuses == {"validated"} else "preliminary_unreviewed"
            ),
            "statuses": sorted(statuses),
            "question_count": len(cases),
            "question_set_sha256": _sha256(question_path),
            "rag_config_sha256": _sha256(rag_path),
            "corpus_sha256": _sha256(rag_config["corpus_path"]),
        },
        "aggregate_metrics": _metrics(rows),
        "per_question": rows,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    """Run the configured answerability experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    csv_path, json_path = run_abstention_experiment(args.config)
    print("CSV:", csv_path, "JSON:", json_path)


if __name__ == "__main__":
    main()
