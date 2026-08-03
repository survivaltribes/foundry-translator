from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from foundry_translator.openai_translator import OpenAITranslatorCountError
from foundry_translator.openai_translator import OpenAITranslator
from foundry_translator.scanner import TranslationEntry


def _build_placeholder_rich_source() -> str:
    return (
        '<p>Hello @Embed[foo] and @Check[skill] and @Damage[1d6] and @Template[spell] '
        'and [[link]] and {{macro}} and @UUID[123e4567-e89b-12d3-a456-426614174000]</p>'
    )


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


def test_duplicate_placeholder_restore_persists_artifacts_and_logs_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "translations": [
                    {
                        "id": 1,
                        "translation": "Bonjour __FT_FAKE_99999__ __FT_FAKE_99999__",
                    }
                ]
            }
        )
    )

    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        batch_size=1,
        client=client,
    )

    debug_dir = tmp_path / "debug" / "restore_duplicate_placeholders_test"

    monkeypatch.setattr(
        translator,
        "_get_restore_debug_artifact_dir",
        lambda: debug_dir,
    )
    caplog.set_level("INFO")

    entries = [
        TranslationEntry(
            file=Path("sample.json"),
            path=["description"],
            field="description",
            source="<p>Hello</p>",
        )
    ]

    with pytest.raises(ValueError, match="Duplicate placeholders detected in protected text"):
        translator.translate(entries)

    assert (debug_dir / "original_source.txt").read_text(encoding="utf-8") == "<p>Hello</p>"
    assert "__FT_HTML_00001__" in (debug_dir / "protected_source.txt").read_text(encoding="utf-8")
    assert (
        (debug_dir / "translated_protected.txt").read_text(encoding="utf-8")
        == "Bonjour __FT_FAKE_99999__ __FT_FAKE_99999__"
    )
    assert "__FT_FAKE_99999__" in (debug_dir / "restored_attempt.txt").read_text(encoding="utf-8")
    placeholders_before = json.loads((debug_dir / "placeholders_before_restore.json").read_text(encoding="utf-8"))
    placeholders_after = json.loads((debug_dir / "placeholders_after_translation.json").read_text(encoding="utf-8"))
    assert "__FT_HTML_00001__" in placeholders_before
    assert "__FT_HTML_00002__" in placeholders_before
    assert placeholders_after.count("__FT_FAKE_99999__") >= 2
    assert (debug_dir / "file_name.txt").read_text(encoding="utf-8") == "sample.json"
    assert (debug_dir / "field_name.txt").read_text(encoding="utf-8") == "description"
    assert (debug_dir / "json_path.txt").read_text(encoding="utf-8") == "$['description']"

    assert any(record.getMessage() == "saved restore duplicate placeholder debug artifacts" for record in caplog.records)
    duplicate_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "duplicate placeholders detected during restore"
    ]
    assert duplicate_logs
    log_record = duplicate_logs[0]
    assert getattr(log_record, "entry_id", None) == 1
    assert getattr(log_record, "field", None) == "description"
    assert getattr(log_record, "file", None) == "sample.json"
    original_placeholders = getattr(log_record, "original_placeholders", None)
    translated_placeholders = getattr(log_record, "translated_placeholders", None)
    assert isinstance(original_placeholders, list)
    assert isinstance(translated_placeholders, list)
    assert "__FT_HTML_00001__" in original_placeholders
    assert translated_placeholders.count("__FT_FAKE_99999__") >= 2
    assert getattr(log_record, "json_path", None) == "$['description']"


def test_strip_appended_original_protected_source_keeps_normal_translation() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
    )

    result = translator._strip_appended_original_protected_source(
        translated_text="Bonjour",
        protected_source="__FT_HTML_00001__Hello__FT_HTML_00002__",
    )

    assert result == "Bonjour"


def test_strip_appended_original_protected_source_strips_exact_suffix() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
    )

    protected_source = "__FT_HTML_00001__Hello__FT_HTML_00002__"
    result = translator._strip_appended_original_protected_source(
        translated_text=f"Bonjour{protected_source}",
        protected_source=protected_source,
    )

    assert result == "Bonjour"


def test_strip_appended_original_protected_source_strips_suffix_starting_at_placeholder_7() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
    )

    protected_text = translator.protector.protect(_build_placeholder_rich_source())
    placeholder_matches = list(re.finditer(r"__FT_[A-Z_]+_\d{5}__", protected_text.protected))
    assert len(placeholder_matches) >= 7
    suffix_start = placeholder_matches[6].start()
    suffix = protected_text.protected[suffix_start:]
    masked_translation = protected_text.protected.replace("Hello", "Bonjour")

    result = translator._strip_appended_original_protected_source(
        translated_text=f"{masked_translation}{suffix}",
        protected_source=protected_text.protected,
    )

    assert result == masked_translation


def test_strip_appended_original_protected_source_strips_suffix_starting_at_arbitrary_placeholder() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
    )

    protected_text = translator.protector.protect(_build_placeholder_rich_source())
    placeholder_matches = list(re.finditer(r"__FT_[A-Z_]+_\d{5}__", protected_text.protected))
    suffix_start = placeholder_matches[3].start()
    suffix = protected_text.protected[suffix_start:]
    masked_translation = protected_text.protected.replace("Hello", "Bonjour")

    result = translator._strip_appended_original_protected_source(
        translated_text=f"{masked_translation}{suffix}",
        protected_source=protected_text.protected,
    )

    assert result == masked_translation


def test_strip_appended_original_protected_source_does_not_strip_partial_overlap() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
    )

    protected_source = "__FT_HTML_00001__Hello__FT_HTML_00002__"
    partial = protected_source[:-1]
    translated = f"Bonjour{partial}"
    result = translator._strip_appended_original_protected_source(
        translated_text=translated,
        protected_source=protected_source,
    )

    assert result == translated


def test_strip_appended_original_protected_source_does_not_strip_legitimate_repeat_structure() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
    )

    protected_text = translator.protector.protect(_build_placeholder_rich_source())
    translated = (
        "Bonjour bonjour. "
        "Le texte se répète, mais chaque placeholder protégé n'apparaît qu'une seule fois: "
        f"{protected_text.protected} "
        "Encore bonjour."
    )

    result = translator._strip_appended_original_protected_source(
        translated_text=translated,
        protected_source=protected_text.protected,
    )

    assert result == translated


def test_strip_appended_original_protected_source_truncates_at_first_repeated_placeholder() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
    )

    protected_text = translator.protector.protect(_build_placeholder_rich_source())
    placeholder_matches = list(re.finditer(r"__FT_[A-Z_]+_\d{5}__", protected_text.protected))
    first_placeholder = placeholder_matches[0].group(0)
    second_placeholder = placeholder_matches[1].group(0)
    translated = f"Bonjour {first_placeholder} texte {second_placeholder} suite {first_placeholder} DUP"

    result = translator._strip_appended_original_protected_source(
        translated_text=translated,
        protected_source=protected_text.protected,
    )

    assert result == f"Bonjour {first_placeholder} texte {second_placeholder} suite "


def test_translate_strips_appended_protected_source_before_restore() -> None:
    client = Mock()
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        batch_size=1,
        client=client,
    )

    source = "<p>Hello</p>"
    protected_source = translator.protector.protect(source).protected
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "translations": [
                    {
                        "id": 1,
                        "translation": f"Bonjour{protected_source}",
                    }
                ]
            }
        )
    )

    entries = [
        TranslationEntry(
            file=Path("sample.json"),
            path=["description"],
            field="description",
            source=source,
        )
    ]

    translated = translator.translate(entries)

    assert translated[0].source == "<p>Bonjour</p>"


def test_replay_restore_from_debug_dir_replays_saved_diagnostics(tmp_path: Path) -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        client=object(),
    )

    source = _build_placeholder_rich_source()
    protected_text = translator.protector.protect(source)
    placeholder_matches = list(re.finditer(r"__FT_[A-Z_]+_\d{5}__", protected_text.protected))
    suffix_start = placeholder_matches[6].start()
    duplicated_suffix = protected_text.protected[suffix_start:]
    masked_translation = protected_text.protected.replace("Hello", "Bonjour")
    translated_protected = f"{masked_translation}{duplicated_suffix}"

    debug_dir = tmp_path / "restore_duplicate_placeholders_test"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "original_source.txt").write_text(source, encoding="utf-8")
    (debug_dir / "protected_source.txt").write_text(protected_text.protected, encoding="utf-8")
    (debug_dir / "translated_protected.txt").write_text(translated_protected, encoding="utf-8")
    (debug_dir / "placeholders_before_restore.json").write_text(
        json.dumps(list(protected_text.placeholders.keys()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (debug_dir / "placeholders_after_translation.json").write_text(
        json.dumps([match.group(0) for match in placeholder_matches], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (debug_dir / "file_name.txt").write_text("andrella.json", encoding="utf-8")
    (debug_dir / "field_name.txt").write_text("description", encoding="utf-8")
    (debug_dir / "json_path.txt").write_text("$['description']", encoding="utf-8")
    (debug_dir / "restored_attempt.txt").write_text(translated_protected, encoding="utf-8")

    replayed = OpenAITranslator.replay_restore_from_debug_dir(debug_dir)
    expected = translator._restore_protected_text(protected_text, masked_translation)

    assert replayed == expected


def test_restore_protected_text_handles_masked_translation_with_appended_suffix() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        client=object(),
    )

    source = _build_placeholder_rich_source()
    protected_text = translator.protector.protect(source)
    masked_translation = protected_text.protected.replace("Hello", "Bonjour")
    placeholder_matches = list(re.finditer(r"__FT_[A-Z_]+_\d{5}__", protected_text.protected))
    suffix_start = placeholder_matches[6].start()
    duplicated_suffix = protected_text.protected[suffix_start:]

    restored = translator._restore_protected_text(
        protected_text,
        f"{masked_translation}{duplicated_suffix}",
    )

    assert "Bonjour" in restored
    assert "__FT_" not in restored
