from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from foundry_translator.openai_translator import OpenAITranslatorCountError
from foundry_translator.openai_translator import OpenAITranslator
from foundry_translator.scanner import TranslationEntry


def test_openai_translator_preserves_order_and_count() -> None:
    client = Mock()
    response = SimpleNamespace(
        output_text=json.dumps(
            {
                "translations": [
                    {"id": 1, "translation": "Bonjour"},
                    {"id": 2, "translation": "Salut"},
                ]
            }
        )
    )
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
        max_prompt_chars=500,
    )

    observed_prompts: list[str] = []

    def fake_call_openai(prompt: str) -> str:
        observed_prompts.append(prompt)
        request_items = translator._extract_input_items_from_prompt(prompt)
        return json.dumps(
            {
                "translations": [
                    {"id": item["id"], "translation": f"translated-{item['id']}"}
                    for item in request_items
                ]
            }
        )

    translator._call_openai = fake_call_openai  # type: ignore[assignment]

    texts = [f"hello-{index}" for index in range(7)]

    translated = translator.translate_batch(
        texts,
        source_language="English",
        target_language="French",
    )

    assert len(translated) == len(texts)
    assert all(len(prompt) <= 500 for prompt in observed_prompts)
    assert len(observed_prompts) >= 1


def test_translate_batches_entry_payloads_by_prompt_size() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        max_prompt_chars=500,
    )

    observed_prompts: list[str] = []

    def fake_call_openai(prompt: str) -> str:
        observed_prompts.append(prompt)
        request_items = translator._extract_input_items_from_prompt(prompt)
        return json.dumps(
            {
                "translations": [
                    {"id": item["id"], "translation": f"translated-{item['id']}"}
                    for item in request_items
                ]
            }
        )

    translator._call_openai = fake_call_openai  # type: ignore[assignment]

    entries = [
        TranslationEntry(file=Path(f"{index}.json"), path=["name"], field="name", source=f"Hello {index}")
        for index in range(7)
    ]

    translated = translator.translate(entries)

    assert len(translated) == len(entries)
    assert all(len(prompt) <= 500 for prompt in observed_prompts)
    assert len(observed_prompts) >= 1


def test_invalid_json_persists_full_prompt_and_response_and_logs_parse_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        max_retries=1,
    )

    full_response = "{not-json"
    prompt_path = tmp_path / "debug" / "failed_prompt.txt"
    response_path = tmp_path / "debug" / "failed_response.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("stale prompt", encoding="utf-8")
    response_path.write_text("stale response", encoding="utf-8")

    def fake_call_openai(prompt: str) -> str:
        return full_response

    monkeypatch.setattr(
        translator,
        "_get_debug_artifact_paths",
        lambda: (prompt_path, response_path),
    )
    translator._call_openai = fake_call_openai  # type: ignore[assignment]
    caplog.set_level("INFO")

    with pytest.raises(OpenAITranslatorCountError, match="Response was not valid JSON"):
        translator.translate_batch(
            ["first", "second"],
            source_language="English",
            target_language="French",
        )

    expected_prompt = translator._build_prompt(
        ["first", "second"],
        source_language="English",
        target_language="French",
        glossary=None,
    )
    assert prompt_path.read_text(encoding="utf-8") == expected_prompt
    assert response_path.read_text(encoding="utf-8") == full_response
    assert any(record.getMessage() == "saved OpenAI response debug artifacts" for record in caplog.records)
    assert any(record.getMessage() == "translation response parse/validation failed" for record in caplog.records)
    assert any(getattr(record, "prompt_path", None) == str(prompt_path.resolve()) for record in caplog.records)
    assert any(getattr(record, "response_path", None) == str(response_path.resolve()) for record in caplog.records)
