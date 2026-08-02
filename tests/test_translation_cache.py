from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from foundry_translator.openai_translator import OpenAITranslator
from foundry_translator.pipeline import Pipeline
from foundry_translator.scanner import TranslationEntry
from foundry_translator.translation_cache import TranslationCache
from foundry_translator.translator import DummyTranslator


def test_translation_cache_persists_entries(tmp_path: Path) -> None:
    cache_path = tmp_path / "translation_cache.json"
    cache = TranslationCache(cache_path)

    assert cache.get("Hello") is None

    cache.put("Hello", "Bonjour")
    cache.save()

    reloaded = TranslationCache(cache_path)
    reloaded.load()

    assert reloaded.get("Hello") == "Bonjour"


def test_translation_cache_ignores_invalid_on_disk_payload(tmp_path: Path) -> None:
    cache_path = tmp_path / "translation_cache.json"
    cache_path.write_text("{not valid json", encoding="utf-8")

    cache = TranslationCache(cache_path)

    assert cache.get("Hello") is None


def test_openai_translator_reuses_cached_translations_without_calling_openai(tmp_path: Path) -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"translations": [{"id": 1, "translation": "Bonjour"}]}'
    )

    cache = TranslationCache(tmp_path / "translation_cache.json")
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        batch_size=2,
        client=client,
        cache=cache,
    )

    entries = [
        TranslationEntry(file=Path("a.json"), path=["name"], field="name", source="Hello"),
        TranslationEntry(file=Path("b.json"), path=["description"], field="description", source="Hello"),
    ]

    translated = translator.translate(entries)

    assert [item.source for item in translated] == ["Bonjour", "Bonjour"]
    client.responses.create.assert_called_once()
    assert cache.get("Hello") == "Bonjour"


def test_pipeline_injects_cache_into_translator(tmp_path: Path) -> None:
    translator = DummyTranslator()
    cache = TranslationCache(tmp_path / "translation_cache.json")

    pipeline = Pipeline(translator=translator, cache=cache)

    assert pipeline.cache is cache
    assert translator.cache is cache
