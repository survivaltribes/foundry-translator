"""Translation abstractions for converting scanner results into translated entries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING

from .scanner import TranslationEntry

if TYPE_CHECKING:
    from .translation_cache import TranslationCache


class Translator(ABC):
    """Abstract base class for translation strategies."""

    def __init__(self, cache: TranslationCache | None = None) -> None:
        self.cache = cache

    @abstractmethod
    def translate(self, entries: list[TranslationEntry]) -> list[TranslationEntry]:
        """Return translated entries from the provided scanner results."""


class DummyTranslator(Translator):
    """Simple translator that leaves the source text unchanged."""

    def translate(self, entries: list[TranslationEntry]) -> list[TranslationEntry]:
        return [replace(entry) for entry in entries]
