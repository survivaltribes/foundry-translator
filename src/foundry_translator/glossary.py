"""Small glossary helper for providing predefined translations."""

from __future__ import annotations


class Glossary:
    """Store source-to-target translations and return them on demand."""

    def __init__(self, entries: dict[str, str] | None = None) -> None:
        self._entries = dict(entries or {})

    def get(self, text: str) -> str | None:
        return self._entries.get(text)

    def put(self, source: str, target: str) -> None:
        self._entries[source] = target
