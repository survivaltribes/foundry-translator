"""Recursive scanner for Babele-style JSON export files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson


@dataclass(slots=True)
class ScanIssue:
    """A non-fatal issue encountered while scanning a JSON file."""

    file: Path
    message: str

TRANSLATABLE_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "description",
        "caption",
        "content",
        "text",
        "label",
        "hint",
    }
)

IGNORED_JSON_FILES: frozenset[str] = frozenset({"module.json", "system.json", "package.json"})


@dataclass(slots=True)
class TranslationEntry:
    """A translatable text found inside a JSON document."""

    file: Path
    path: list[str | int]
    field: str
    source: str


@dataclass(slots=True)
class JsonDocument:
    """A JSON document loaded from disk for inspection."""

    path: Path
    data: Any


class Scanner:
    """Recursively scan exported Babele JSON files for translatable content."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()

        if not self.root.exists():
            raise FileNotFoundError(f"Scan root does not exist: {self.root}")

    def discover(self) -> list[Path]:
        """Return all JSON files under the scan root, excluding common manifest files."""

        files = [
            path
            for path in sorted(self.root.rglob("*.json"))
            if path.is_file() and path.name not in IGNORED_JSON_FILES
        ]
        return files

    def load(self, file: Path) -> JsonDocument:
        """Load a JSON document from disk with orjson."""

        try:
            with file.open("rb") as handle:
                data = orjson.loads(handle.read())
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"JSON file not found: {file}") from exc
        except orjson.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON content in {file}: {exc}") from exc

        return JsonDocument(path=file, data=data)

    def scan(self) -> tuple[list[JsonDocument], list[TranslationEntry], list[ScanIssue]]:
        """Scan the root and return loaded documents, entries, and any non-fatal issues."""

        documents: list[JsonDocument] = []
        entries: list[TranslationEntry] = []
        issues: list[ScanIssue] = []

        for file in self.discover():
            try:
                document = self.load(file)
            except (ValueError, FileNotFoundError) as exc:
                issues.append(ScanIssue(file=file, message=str(exc)))
                continue

            documents.append(document)
            entries.extend(self._walk(document.data, [], file))

        return documents, entries, issues

    def _walk(self, obj: Any, path: list[str | int], file: Path) -> list[TranslationEntry]:
        """Recursively inspect JSON-like values and collect translatable strings."""

        if isinstance(obj, dict):
            results: list[TranslationEntry] = []
            for key, value in obj.items():
                current_path = [*path, key]

                if key in TRANSLATABLE_FIELDS and isinstance(value, str) and value.strip():
                    results.append(
                        TranslationEntry(
                            file=file,
                            path=current_path,
                            field=key,
                            source=value,
                        )
                    )

                results.extend(self._walk(value, current_path, file))
            return results

        if isinstance(obj, list):
            results = []
            for index, value in enumerate(obj):
                results.extend(self._walk(value, [*path, index], file))
            return results

        return []
