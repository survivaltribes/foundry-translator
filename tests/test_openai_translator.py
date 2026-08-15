from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from foundry_translator.openai_translator import OpenAITranslatorCountError
from foundry_translator.openai_translator import PlaceholderMismatchError
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


def test_translate_batch_batches_by_max_items() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        batch_size=2,
        max_prompt_chars=10000,
    )

    observed_batch_sizes: list[int] = []

    def fake_call_openai(prompt: str) -> str:
        request_items = translator._extract_input_items_from_prompt(prompt)
        observed_batch_sizes.append(len(request_items))
        return json.dumps(
            {
                "translations": [
                    {"id": item["id"], "translation": f"translated-{item['id']}"}
                    for item in request_items
                ]
            }
        )

    translator._call_openai = fake_call_openai  # type: ignore[assignment]

    texts = [f"hello-{index}" for index in range(5)]
    translated = translator.translate_batch(
        texts,
        source_language="English",
        target_language="French",
    )

    assert len(translated) == len(texts)
    assert observed_batch_sizes == [2, 2, 1]


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


def test_translate_batches_entry_payloads_by_max_items() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        batch_size=2,
        max_prompt_chars=10000,
    )

    observed_batch_sizes: list[int] = []

    def fake_call_openai(prompt: str) -> str:
        request_items = translator._extract_input_items_from_prompt(prompt)
        observed_batch_sizes.append(len(request_items))
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
        for index in range(5)
    ]

    translated = translator.translate(entries)

    assert len(translated) == len(entries)
    assert observed_batch_sizes == [2, 2, 1]


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


def test_placeholder_mismatch_persists_artifacts_and_logs_context(
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

    with pytest.raises(PlaceholderMismatchError, match="Placeholder mismatch after translation"):
        translator.translate(entries)

    assert (debug_dir / "original_source.txt").read_text(encoding="utf-8") == "<p>Hello</p>"
    assert "__FT_HTML_00001__" in (debug_dir / "protected_source.txt").read_text(encoding="utf-8")
    assert (
        (debug_dir / "translated_protected.txt").read_text(encoding="utf-8")
        == "Bonjour __FT_FAKE_99999__ __FT_FAKE_99999__"
    )
    assert (debug_dir / "sanitized_translated.txt").read_text(encoding="utf-8") == "Bonjour __FT_FAKE_99999__ __FT_FAKE_99999__"
    assert (debug_dir / "exception.txt").read_text(encoding="utf-8").startswith("Placeholder mismatch after translation")
    assert "__FT_FAKE_99999__" in (debug_dir / "restored_attempt.txt").read_text(encoding="utf-8")
    placeholders_before = json.loads((debug_dir / "placeholders_before_restore.json").read_text(encoding="utf-8"))
    placeholders_after = json.loads((debug_dir / "placeholders_after_restore.json").read_text(encoding="utf-8"))
    assert "__FT_HTML_00001__" in placeholders_before
    assert "__FT_HTML_00002__" in placeholders_before
    assert placeholders_after.count("__FT_FAKE_99999__") >= 2
    assert (debug_dir / "file_name.txt").read_text(encoding="utf-8") == "sample.json"
    assert (debug_dir / "field_name.txt").read_text(encoding="utf-8") == "description"
    assert (debug_dir / "json_path.txt").read_text(encoding="utf-8") == "$['description']"
    assert client.responses.create.call_count == 2

    assert any(record.getMessage() == "saved restore debug artifacts" for record in caplog.records)
    mismatch_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "placeholder mismatch persisted after retry"
    ]
    assert mismatch_logs
    log_record = mismatch_logs[0]
    assert getattr(log_record, "entry_id", None) == 1
    assert getattr(log_record, "field", None) == "description"
    assert getattr(log_record, "file", None) == "sample.json"
    missing_placeholders = getattr(log_record, "missing_placeholders", None)
    unexpected_placeholders = getattr(log_record, "unexpected_placeholders", None)
    assert isinstance(missing_placeholders, list)
    assert isinstance(unexpected_placeholders, list)
    assert "__FT_HTML_00001__" in missing_placeholders
    assert "__FT_FAKE_99999__" in unexpected_placeholders
    assert getattr(log_record, "json_path", None) == "$['description']"


def test_non_duplicate_restore_value_error_persists_replayable_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = Mock()
    # Provide only one known placeholder so validation raises a mismatch before restore.
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "translations": [
                    {
                        "id": 1,
                        "translation": "__FT_HTML_00001__Bonjour",
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

    debug_dir = tmp_path / "debug" / "restore_unexpected_placeholders_test"
    monkeypatch.setattr(
        translator,
        "_get_restore_debug_artifact_dir",
        lambda: debug_dir,
    )

    entries = [
        TranslationEntry(
            file=Path("sample.json"),
            path=["description"],
            field="description",
            source="<p>Hello</p>",
        )
    ]

    with pytest.raises(PlaceholderMismatchError, match="Placeholder mismatch after translation"):
        translator.translate(entries)

    assert (debug_dir / "exception.txt").read_text(encoding="utf-8").startswith("Placeholder mismatch after translation")
    assert (debug_dir / "original_source.txt").read_text(encoding="utf-8") == "<p>Hello</p>"
    assert (debug_dir / "translated_protected.txt").read_text(encoding="utf-8") == "__FT_HTML_00001__Bonjour"
    assert (debug_dir / "sanitized_translated.txt").read_text(encoding="utf-8") == "__FT_HTML_00001__Bonjour"
    assert (debug_dir / "restored_attempt.txt").read_text(encoding="utf-8") == "__FT_HTML_00001__Bonjour"

    placeholders_before = json.loads((debug_dir / "placeholders_before_restore.json").read_text(encoding="utf-8"))
    placeholders_after = json.loads((debug_dir / "placeholders_after_restore.json").read_text(encoding="utf-8"))
    assert "__FT_HTML_00001__" in placeholders_before
    assert "__FT_HTML_00002__" in placeholders_before
    assert placeholders_after == ["__FT_HTML_00001__"]
    assert client.responses.create.call_count == 2


def test_placeholder_mismatch_retry_once_then_success() -> None:
    client = Mock()
    client.responses.create.side_effect = [
        SimpleNamespace(
            output_text=json.dumps(
                {
                    "translations": [
                        {
                            "id": 1,
                            "translation": "__FT_HTML_00001__Bonjour",
                        }
                    ]
                }
            )
        ),
        SimpleNamespace(
            output_text=json.dumps(
                {
                    "translations": [
                        {
                            "id": 1,
                            "translation": "__FT_HTML_00001__Bonjour__FT_HTML_00002__",
                        }
                    ]
                }
            )
        ),
    ]

    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        batch_size=1,
        client=client,
    )

    entries = [
        TranslationEntry(
            file=Path("sample.json"),
            path=["description"],
            field="description",
            source="<p>Hello</p>",
        )
    ]

    translated = translator.translate(entries)

    assert translated[0].source == "<p>Bonjour</p>"
    assert client.responses.create.call_count == 2


def test_placeholder_mismatch_retries_only_failing_entries() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        batch_size=2,
        client=object(),
    )

    entries = [
        TranslationEntry(
            file=Path("a.json"),
            path=["description"],
            field="description",
            source="<p>One</p>",
        ),
        TranslationEntry(
            file=Path("b.json"),
            path=["description"],
            field="description",
            source="<p>Two</p>",
        ),
    ]

    protected_one = translator.protector.protect(entries[0].source).protected
    protected_two = translator.protector.protect(entries[1].source).protected

    observed_requests: list[list[str]] = []

    def fake_translate_batch(
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
        glossary: dict[str, str] | None = None,
    ) -> list[str]:
        observed_requests.append(texts)
        if len(observed_requests) == 1:
            assert texts == [protected_one, protected_two]
            return [
                "__FT_HTML_00001__Un__FT_HTML_00002__",
                "__FT_HTML_00001__Deux",
            ]
        assert len(observed_requests) == 2
        assert texts == [protected_two]
        return ["__FT_HTML_00001__Deux__FT_HTML_00002__"]

    translator._translate_batch = fake_translate_batch  # type: ignore[assignment]

    translated = translator.translate(entries)

    assert [item.source for item in translated] == ["<p>Un</p>", "<p>Deux</p>"]
    assert len(observed_requests) == 2
    assert observed_requests[0] == [protected_one, protected_two]
    assert observed_requests[1] == [protected_two]


def test_e1_tolerates_missing_empty_strong_pair_with_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = Mock()
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        batch_size=1,
        client=client,
    )

    source = (
        "A character's<strong> </strong>and a creature's options are listed in "
        "@UUID[JournalEntry.example.JournalEntryPage.sample]{Actions}."
    )
    protected_text = translator.protector.protect(source)

    strong_placeholders = {
        placeholder_name
        for placeholder_name, placeholder in protected_text.placeholders.items()
        if hasattr(placeholder, "original") and placeholder.original in {"<strong>", "</strong>"}
    }
    assert len(strong_placeholders) == 2

    non_html_placeholders = [
        placeholder_name
        for placeholder_name, placeholder in protected_text.placeholders.items()
        if hasattr(placeholder, "category") and placeholder.category != "HTML"
    ]
    assert non_html_placeholders
    kept_non_html_placeholder = non_html_placeholders[0]

    # Keep at least one non-HTML placeholder in the model output on purpose:
    # if all placeholders disappear, restore can follow a different reconstruction
    # path and this no longer reproduces the real incident.
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "translations": [
                    {
                        "id": 1,
                        "translation": (
                            "La fiche de personnage et celle de creature listent les options "
                            f"{kept_non_html_placeholder}."
                        ),
                    }
                ]
            }
        )
    )

    debug_dir = tmp_path / "debug" / "restore_duplicate_placeholders_characterization"
    monkeypatch.setattr(
        translator,
        "_get_restore_debug_artifact_dir",
        lambda: debug_dir,
    )

    entries = [
        TranslationEntry(
            file=Path("sample.json"),
            path=["description"],
            field="description",
            source=source,
        )
    ]

    caplog.set_level("WARNING")
    translated = translator.translate(entries)

    exception_path = debug_dir / "exception.txt"

    assert len(translated) == 1
    assert "<strong>" not in translated[0].source
    assert exception_path.exists() is False
    assert "@UUID[" in translated[0].source

    warning_records = [
        record
        for record in caplog.records
        if record.getMessage() == "tolerated missing placeholders; continuing translation"
    ]
    assert warning_records
    warning = warning_records[0]
    assert getattr(warning, "rule", None) == "E1_TOLERATED_MISSING"
    assert getattr(warning, "severity", None) == "WARNING"
    assert set(getattr(warning, "placeholders", [])) == strong_placeholders
    assert getattr(warning, "pipeline_continues", None) is True


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


def test_sanitize_translated_protected_text_keeps_original_when_strip_would_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
    )

    protected_source = "__FT_HTML_00001__Hello__FT_HTML_00002__"
    caplog.set_level("WARNING")

    sanitized = translator._sanitize_translated_protected_text(
        translated_text=protected_source,
        protected_source=protected_source,
    )

    assert sanitized == protected_source
    assert any(
        record.getMessage() == "sanitization would remove entire translated text; keeping original"
        for record in caplog.records
    )


def test_sanitize_translated_protected_text_rejects_non_duplicated_placeholder_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
    )

    monkeypatch.setattr(
        translator,
        "_strip_appended_original_protected_source",
        lambda *, translated_text, protected_source: "Bonjour__FT_HTML_00002__",
    )

    with pytest.raises(AssertionError, match="non-suffix segment|non-duplicated placeholder"):
        translator._sanitize_translated_protected_text(
            translated_text="__FT_HTML_00001__Bonjour__FT_HTML_00002__",
            protected_source="__FT_HTML_00001__Hello__FT_HTML_00002__",
        )


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


def test_plain_translation_without_placeholders_is_reconstructed_deterministically() -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        client=object(),
    )

    source = "<p>Hello</p>"
    protected_text = translator.protector.protect(source)

    sanitized, restored_attempt, should_restore = translator._prepare_restore_attempt(
        protected_text,
        "Bonjour",
    )

    assert sanitized == "Bonjour"
    assert should_restore is True
    assert restored_attempt == protected_text.protected.replace("Hello", "Bonjour")
    assert translator._restore_protected_text(protected_text, "Bonjour") == "<p>Bonjour</p>"


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


def test_replay_restore_from_debug_dir_replays_saved_restore_failure(tmp_path: Path) -> None:
    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        client=object(),
    )

    source = "<p>Hello</p>"
    protected_text = translator.protector.protect(source)

    debug_dir = tmp_path / "restore_failure_test"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "exception.txt").write_text(
        "Missing placeholders: ['__FT_HTML_00002__']",
        encoding="utf-8",
    )
    (debug_dir / "original_source.txt").write_text(source, encoding="utf-8")
    (debug_dir / "protected_source.txt").write_text(protected_text.protected, encoding="utf-8")
    (debug_dir / "translated_protected.txt").write_text("__FT_HTML_00001__Bonjour", encoding="utf-8")
    (debug_dir / "sanitized_translated.txt").write_text("__FT_HTML_00001__Bonjour", encoding="utf-8")
    (debug_dir / "restored_attempt.txt").write_text("__FT_HTML_00001__Bonjour", encoding="utf-8")
    (debug_dir / "placeholders_before_restore.json").write_text(
        json.dumps(list(protected_text.placeholders.keys()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (debug_dir / "placeholders_after_restore.json").write_text(
        json.dumps(["__FT_HTML_00001__"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (debug_dir / "file_name.txt").write_text("sample.json", encoding="utf-8")
    (debug_dir / "field_name.txt").write_text("description", encoding="utf-8")
    (debug_dir / "json_path.txt").write_text("$['description']", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing placeholders"):
        OpenAITranslator.replay_restore_from_debug_dir(debug_dir)


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
