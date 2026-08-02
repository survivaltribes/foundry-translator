from __future__ import annotations

import json
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


def test_translate_batch_batches_by_prompt_size() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        max_prompt_chars=220,
    )

    observed_prompts: list[str] = []

    def fake_call_openai(prompt: str) -> str:
        observed_prompts.append(prompt)
        batch_size = len([line for line in prompt.splitlines() if line[:1].isdigit()])
        return json.dumps([f"translated-{index}" for index in range(batch_size)])

    translator._call_openai = fake_call_openai  # type: ignore[assignment]

    texts = [f"hello-{index}" for index in range(7)]

    translated = translator.translate_batch(
        texts,
        source_language="English",
        target_language="French",
    )

    assert len(translated) == len(texts)
    assert all(len(prompt) <= 220 for prompt in observed_prompts)
    assert len(observed_prompts) >= 2


def test_translate_batches_entry_payloads_by_prompt_size() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        max_prompt_chars=220,
    )

    observed_prompts: list[str] = []

    def fake_call_openai(prompt: str) -> str:
        observed_prompts.append(prompt)
        batch_size = len([line for line in prompt.splitlines() if line[:1].isdigit()])
        return json.dumps([f"translated-{index}" for index in range(batch_size)])

    translator._call_openai = fake_call_openai  # type: ignore[assignment]

    entries = [
        TranslationEntry(file=Path(f"{index}.json"), path=["name"], field="name", source=f"Hello {index}")
        for index in range(7)
    ]

    translated = translator.translate(entries)

    assert len(translated) == len(entries)
    assert all(len(prompt) <= 220 for prompt in observed_prompts)
    assert len(observed_prompts) >= 2
