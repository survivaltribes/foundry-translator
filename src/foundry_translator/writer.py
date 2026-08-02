"""Utilities for writing translated JSON back to disk without changing structure."""

from __future__ import annotations

import re
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

        self._validate_payload(payload)

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

    def _validate_payload(self, payload: Any) -> None:
        serialized = orjson.dumps(payload)
        if not serialized:
            raise ValueError("Translated JSON is empty")

        text = serialized.decode("utf-8")
        if "__FOUNDRY_PLACEHOLDER_" in text:
            raise ValueError("Found unresolved placeholders in translated JSON")

        if self._contains_invalid_macros(text):
            raise ValueError("Found invalid Foundry macros in translated JSON")

    def _contains_invalid_macros(self, text: str) -> bool:
        pattern = re.compile(r"\{\{[^{}]+\}\}|\[\[[^\]]+\]\]")
        matches = list(pattern.finditer(text))
        return any(not self._looks_like_valid_macro(match.group(0)) for match in matches)

    def _looks_like_valid_macro(self, value: str) -> bool:
        if value.startswith("{{") and value.endswith("}}"):
            return bool(value[2:-2].strip())
        if value.startswith("[[") and value.endswith("]]"):
            return bool(value[2:-2].strip())
        return False
