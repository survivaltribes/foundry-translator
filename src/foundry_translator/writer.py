"""Utilities for writing translated JSON back to disk without changing structure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from .scanner import TranslationEntry


class JsonWriter:
    """Write translated content back into a JSON document using scanner paths."""

    def __init__(self, *, indent: int = 2) -> None:
        self.indent = indent

    def load(self, file: Path) -> Any:
        with file.open("rb") as handle:
            return orjson.loads(handle.read())

    def write(self, source_file: Path, entries: list[TranslationEntry]) -> Path:
        payload = self.load(source_file)

        for entry in entries:
            self._apply_translation(payload, entry.path, entry.source)

        destination = source_file.with_suffix(".translated.json")
        with destination.open("wb") as handle:
            handle.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

        return destination

    def _apply_translation(self, payload: Any, path: list[str | int], value: str) -> None:
        if not path:
            return

        current = payload
        for segment in path[:-1]:
            if isinstance(current, dict):
                if segment not in current:
                    return
                current = current[segment]
            elif isinstance(current, list):
                if not isinstance(segment, int) or segment >= len(current):
                    return
                current = current[segment]
            else:
                return

        last_segment = path[-1]
        if isinstance(current, dict):
            if last_segment in current:
                current[last_segment] = value
        elif isinstance(current, list):
            if isinstance(last_segment, int) and last_segment < len(current):
                current[last_segment] = value
