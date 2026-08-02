from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from foundry_translator.openai_translator import OpenAITranslator
from foundry_translator.scanner import TranslationEntry


def test_openai_translator_preserves_order_and_count() -> None:
    client = Mock()
    response = SimpleNamespace(output_text="Bonjour\nSalut")
    client.responses.create.return_value = response

    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        batch_size=2,
        client=client,
    )

    entries = [
        TranslationEntry(file=Path("a.json"), path=["name"], field="name", source="Hello"),
        TranslationEntry(file=Path("b.json"), path=["description"], field="description", source="World"),
    ]

    translated = translator.translate(entries)

    assert [item.source for item in translated] == ["Bonjour", "Salut"]
    assert [item.path for item in translated] == [entry.path for entry in entries]
    assert [item.field for item in translated] == [entry.field for entry in entries]
    assert len(translated) == len(entries)
