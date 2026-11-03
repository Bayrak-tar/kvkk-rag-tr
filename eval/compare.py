"""Çalıştırılmış experiment JSON dosyalarından tablo ve SVG grafik üretir."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Aggregate metric içeren sonuç JSON'larını yükler ve id'ye göre sıralar."""
    results = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "metadata" in data and "aggregate_metrics" in data:
            data["_path"] = str(path)
            results.append(data)
    return results


def _metric_names(results: list[dict[str, Any]]) -> list[str]:
    """Bütün sonuçlarda bulunan retrieval ve üretim metriklerini adlarıyla seçer."""
    common = set.intersection(
        *(set(item["aggregate_metrics"]) for item in results)
    ) if results else set()
    preferred = [name for name in ("recall@5", "mrr@5", "ndcg@5", "citation_coverage") if name in common]
    if preferred:
        return preferred
    return sorted(common)


def _write_csv(results: list[dict[str, Any]], path: Path, metrics: list[str]) -> None:
    columns = ["experiment_id", "strategy", "embedding_model", "chunking_strategy", *metrics]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for item in results:
            metadata = item["metadata"]
            writer.writerow(
                {
                    "experiment_id": metadata.get("experiment_id", Path(item["_path"]).stem),
                    "strategy": metadata.get("retrieval_strategy", "unknown"),
                    "embedding_model": metadata.get("embedding_model") or "none",
                    "chunking_strategy": metadata.get("chunking_strategy") or "unknown",
                    **{metric: item["aggregate_metrics"].get(metric, "") for metric in metrics},
                }
            )


def _write_markdown(results: list[dict[str, Any]], path: Path, metrics: list[str]) -> None:
    headers = ["Deney", "Arama", "Embedding", "Chunking", "Soru sayısı", *metrics]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for item in results:
        metadata = item["metadata"]
        cells = [
            str(metadata.get("experiment_id", Path(item["_path"]).stem)),
            str(metadata.get("retrieval_strategy", "unknown")),
            str(metadata.get("embedding_model") or "—"),
            str(metadata.get("chunking_strategy") or "unknown"),
            str(metadata.get("question_count", "?")),
            *[
                f"{item['aggregate_metrics'][metric]:.4f}"
                if isinstance(item["aggregate_metrics"].get(metric), (int, float))
                else "—"
                for metric in metrics
            ],
        ]
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_svg(results: list[dict[str, Any]], path: Path, metrics: list[str]) -> None:
    width, height = 980, max(320, 120 + 80 * len(results))
    left, right, top = 250, 50, 45
    plot_width = width - left - right
    row_height = (height - top - 35) / max(1, len(results))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.label{font-size:13px}.title{font-size:18px;font-weight:bold}</style>',
        '<text x="24" y="28" class="title">Retrieval deneyleri — metrikler 0–1</text>',
    ]
    colors = ["#276EF1", "#11A683", "#F2A900", "#A855F7"]
    for metric_index, metric in enumerate(metrics):
        x = left + metric_index * 160
        parts.append(
            f'<rect x="{x}" y="36" width="12" height="12" fill="{colors[metric_index % len(colors)]}"/>'
        )
        parts.append(f'<text x="{x + 18}" y="47" class="label">{html.escape(metric)}</text>')
    for row_index, item in enumerate(results):
        y = top + row_index * row_height
        metadata = item["metadata"]
        label = str(metadata.get("experiment_id", Path(item["_path"]).stem))
        parts.append(f'<text x="20" y="{y + 20:.1f}" class="label">{html.escape(label)}</text>')
        for metric_index, metric in enumerate(metrics):
            value = item["aggregate_metrics"].get(metric)
            if not isinstance(value, (int, float)):
                continue
            bar_y = y + metric_index * 15
            bar_width = max(0.0, min(1.0, float(value))) * plot_width
            color = colors[metric_index % len(colors)]
            parts.append(
                f'<rect x="{left}" y="{bar_y:.1f}" width="{bar_width:.2f}" height="10" rx="3" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{left + bar_width + 6:.1f}" y="{bar_y + 9:.1f}" class="label">{float(value):.3f}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def compare(results_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """CSV, Markdown ve SVG karşılaştırmasını yalnız mevcut sonuçlardan üretir."""
    results = _load_results(results_dir)
    if not results:
        raise ValueError(f"{results_dir} altında karşılaştırılabilir experiment JSON bulunamadı.")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _metric_names(results)
    csv_path = output_dir / "comparison.csv"
    markdown_path = output_dir / "comparison.md"
    svg_path = output_dir / "comparison.svg"
    _write_csv(results, csv_path, metrics)
    _write_markdown(results, markdown_path, metrics)
    _write_svg(results, svg_path, metrics)
    return csv_path, markdown_path, svg_path


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/comparison"))
    args = parser.parse_args()
    paths = compare(args.results_dir, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
