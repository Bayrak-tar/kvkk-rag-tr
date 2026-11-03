"""Summarize human answer annotations without scoring blank fields."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

SCORE_FIELDS = (
    "correctness_1_to_5",
    "completeness_1_to_5",
    "faithfulness_1_to_5",
    "citation_support_1_to_5",
)


def summarize_reviews(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Compute means only for completed 1-5 human ratings."""
    with input_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    scores: dict[str, list[int]] = {field: [] for field in SCORE_FIELDS}
    abstention: list[bool] = []
    for row_number, row in enumerate(rows, 2):
        for field in SCORE_FIELDS:
            value = row.get(field, "").strip()
            if not value:
                continue
            try:
                score = int(value)
            except ValueError as error:
                raise ValueError(f"Row {row_number}: {field} must be an integer 1-5.") from error
            if score not in range(1, 6):
                raise ValueError(f"Row {row_number}: {field} must be an integer 1-5.")
            scores[field].append(score)
        expected = row.get("abstention_expected", "").strip().casefold()
        correct = row.get("abstention_correct", "").strip().casefold()
        if expected in {"true", "false"} and correct in {"true", "false"}:
            abstention.append(expected == correct)

    summary: dict[str, Any] = {
        "reviewed_row_count": len(rows),
        "annotated_score_counts": {field: len(values) for field, values in scores.items()},
        "mean_scores_1_to_5": {
            field: round(mean(values), 4) if values else None
            for field, values in scores.items()
        },
        "abstention_labeled_count": len(abstention),
        "abstention_accuracy": round(sum(abstention) / len(abstention), 4)
        if abstention
        else None,
        "input_file": str(input_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Write aggregate human-review metrics as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_reviews(args.input_csv, args.output_json), ensure_ascii=False))


if __name__ == "__main__":
    main()
