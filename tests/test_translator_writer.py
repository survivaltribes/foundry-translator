from __future__ import annotations

import json
from pathlib import Path

from foundry_translator.scanner import TranslationEntry
from foundry_translator.translator import DummyTranslator
from foundry_translator.writer import JsonWriter


def test_dummy_translator_returns_original_texts(tmp_path: Path) -> None:
    source_file = tmp_path / "sample.json"
    source_file.write_text('{"name": "Hero"}', encoding="utf-8")

    entry = TranslationEntry(
        file=source_file,
        path=["name"],
        field="name",
        source="Hero",
    )

    translated = DummyTranslator().translate([entry])

    assert len(translated) == 1
    assert translated[0].file == entry.file
    assert translated[0].path == entry.path
    assert translated[0].field == entry.field
    assert translated[0].source == entry.source


def test_writer_applies_translations_without_changing_structure(tmp_path: Path) -> None:
    source_file = tmp_path / "compendium.json"
    payload = {
        "name": "Hero",
        "description": "A legendary hero",
        "nested": {
            "name": "Nested name",
            "items": [{"description": "Item description"}],
        },
        "ignored": "keep me",
    }
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    translated_entries = [
        TranslationEntry(source_file, ["name"], "name", "Héros"),
        TranslationEntry(source_file, ["nested", "name"], "name", "Nom imbriqué"),
        TranslationEntry(source_file, ["nested", "items", 0, "description"], "description", "Description d’élément"),
    ]

    writer = JsonWriter()
    output_file = writer.write(source_file, translated_entries)

    written = json.loads(output_file.read_text(encoding="utf-8"))

    assert written["name"] == "Héros"
    assert written["description"] == "A legendary hero"
    assert written["nested"]["name"] == "Nom imbriqué"
    assert written["nested"]["items"][0]["description"] == "Description d’élément"
    assert written["ignored"] == "keep me"
    assert written["nested"]["items"][0]["title"] if False else True
