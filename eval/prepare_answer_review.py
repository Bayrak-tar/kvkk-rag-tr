"""Generate a human-review CSV from a completed RAG generation experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REVIEW_COLUMNS = [
    "question_id",
    "experiment_id",
    "question",
    "reference_answer",
    "gold_article_ids",
    "retrieved_article_ids",
    "answer",
    "latency_ms",
    "citations",
    "verification_status",
    "correctness_1_to_5",
    "completeness_1_to_5",
    "faithfulness_1_to_5",
    "citation_support_1_to_5",
    "abstention_expected",
    "abstention_correct",
    "error_category",
    "severity",
    "reviewer",
    "reviewed_at",
    "notes",
]


def prepare_review(input_path: Path, output_path: Path) -> int:
    """Copy experiment evidence into a blank, manually annotatable CSV."""
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No experiment rows found in {input_path}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in REVIEW_COLUMNS})
    return len(rows)


def main() -> None:
    """Create the review sheet from one result CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    count = prepare_review(args.input_csv, args.output_csv)
    print(f"Prepared {count} rows: {args.output_csv}")


if __name__ == "__main__":
    main()
