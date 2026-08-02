"""Translation abstractions for converting scanner results into translated entries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from .scanner import TranslationEntry


class Translator(ABC):
    """Abstract base class for translation strategies."""

    @abstractmethod
    def translate(self, entries: list[TranslationEntry]) -> list[TranslationEntry]:
        """Return translated entries from the provided scanner results."""


class DummyTranslator(Translator):
    """Simple translator that leaves the source text unchanged.

    This is intended as a placeholder for future integrations such as OpenAITranslator.
    """

    def translate(self, entries: list[TranslationEntry]) -> list[TranslationEntry]:
        return [replace(entry) for entry in entries]
