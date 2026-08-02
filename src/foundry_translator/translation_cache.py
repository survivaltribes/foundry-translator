"""Simple on-disk translation cache for reusing previously translated text."""

from __future__ import annotations

import json
from pathlib import Path


class TranslationCache:
    """Store translations keyed by the original source text."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or "translation_cache.json")
        self._data: dict[str, str] = {}
        self.load()

    def get(self, text: str) -> str | None:
        return self._data.get(text)

    def put(self, source: str, target: str) -> None:
        self._data[source] = target

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._data = {}
            return

        if isinstance(raw, dict):
            self._data = {str(key): str(value) for key, value in raw.items()}
        else:
            self._data = {}
