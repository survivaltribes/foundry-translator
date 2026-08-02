from __future__ import annotations

import json
from pathlib import Path

from foundry_translator.scanner import Scanner, TranslationEntry


def test_scan_finds_only_translatable_fields(tmp_path: Path) -> None:
    payload = {
        "name": "Hero",
        "description": "A legendary hero",
        "caption": "Caption text",
        "label": "Label text",
        "hint": "Hint text",
        "content": "Content text",
        "text": "Text text",
        "ignored": "Should not be captured",
        "nested": {
            "name": "Nested name",
            "notes": "Ignored nested note",
            "items": [
                {"description": "Item description"},
                {"title": "Nope"},
            ],
        },
    }

    sample_file = tmp_path / "compendium.json"
    sample_file.write_text(json.dumps(payload), encoding="utf-8")

    scanner = Scanner(tmp_path)
    documents, entries, issues = scanner.scan()

    assert len(documents) == 1
    assert issues == []
    assert len(entries) == 9

    paths = {(tuple(entry.path), entry.field, entry.source) for entry in entries}
    assert (('name',), 'name', 'Hero') in paths
    assert (('nested', 'name'), 'name', 'Nested name') in paths
    assert (('nested', 'items', 0, 'description'), 'description', 'Item description') in paths
    assert all(entry.file == sample_file for entry in entries)
    assert all(isinstance(entry, TranslationEntry) for entry in entries)
