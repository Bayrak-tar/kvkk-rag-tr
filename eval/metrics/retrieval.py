"""Retrieval için Recall@k, MRR ve binary nDCG metrikleri."""

from __future__ import annotations

import math

from src.retrieval.store import VectorHit


def _article_keys(hits: list[VectorHit]) -> list[str]:
    """Sıralı chunk listesini tekrar etmeyen source/article kimliklerine indirger."""
    keys: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        source_id = str(hit.metadata.get("source_id", ""))
        article_value = hit.metadata.get("article_no", "")
        articles = article_value if isinstance(article_value, list) else str(article_value).split(",")
        for article in articles:
            if source_id and str(article).strip():
                key = f"{source_id}:{str(article).strip()}"
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
    return keys


def retrieval_metrics(
    hits: list[VectorHit],
    gold_article_ids: list[str],
    *,
    k: int,
) -> dict[str, float]:
    """Bir sorunun top-k sonuçları için Recall, reciprocal rank ve nDCG hesaplar."""
    if k <= 0:
        raise ValueError("k pozitif olmalı.")
    if not gold_article_ids:
        raise ValueError("En az bir gold madde kimliği gereklidir.")
    gold = set(gold_article_ids)
    ranked = _article_keys(hits)[:k]
    relevant = [int(article_id in gold) for article_id in ranked]
    recall = sum(relevant) / len(gold)
    reciprocal_rank = next(
        (1.0 / rank for rank, is_relevant in enumerate(relevant, 1) if is_relevant),
        0.0,
    )
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(relevant, 1))
    ideal_relevant = min(k, len(gold))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return {
        f"recall@{k}": recall,
        f"mrr@{k}": reciprocal_rank,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Soru başına metrikleri aritmetik ortalamayla birleştirir."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
