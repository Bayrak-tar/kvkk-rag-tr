"""Metin embedding sağlayıcıları için ortak arayüz."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Sorgu ve mevzuat metinlerini vektörlere dönüştüren arayüz."""

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Sorgu metinlerini vektörlere dönüştürür."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Mevzuat parçalarını vektörlere dönüştürür."""
