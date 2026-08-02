"""Backward-compatible import surface for the shared protection module."""

from __future__ import annotations

from .protect import MarkupProtector, ProtectedText

__all__ = ["MarkupProtector", "ProtectedText"]
