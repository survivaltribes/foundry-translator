from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from foundry_translator.openai_translator import (
    OpenAITranslator,
    OpenAITranslatorCountError,
    OpenAITranslatorRequestError,
)


def test_translate_batch_preserves_order_and_count() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "translations": [
                    {"id": 1, "translation": "Bonjour"},
                    {"id": 2, "translation": "Salut"},
                ]
            }
        )
    )

    translator = OpenAITranslator(api_key="test-key", model="gpt-5.5", batch_size=2, client=client)

    result = translator.translate_batch(
        ["Hello", "World"],
        source_language="English",
        target_language="French",
    )

    assert result == ["Bonjour", "Salut"]
    client.responses.create.assert_called_once()


def test_translate_batch_retries_and_raises_on_count_mismatch() -> None:
    client = Mock()
    client.responses.create.side_effect = [
        SimpleNamespace(output_text='{"translations": [{"id": 1, "translation": "Bonjour"}]}'),
        SimpleNamespace(
            output_text='{"translations": [{"id": 1, "translation": "Bonjour"}, {"id": 2, "translation": "Salut"}]}'
        ),
    ]

    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-5.5",
        batch_size=2,
        max_retries=2,
        retry_delay=0,
        client=client,
    )

    result = translator.translate_batch(
        ["Hello", "World"],
        source_language="English",
        target_language="French",
    )

    assert result == ["Bonjour", "Salut"]
    assert client.responses.create.call_count == 2


def test_translate_batch_raises_on_invalid_response_count() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"translations": [{"id": 1, "translation": "Bonjour"}]}'
    )

    translator = OpenAITranslator(api_key="test-key", model="gpt-5.5", batch_size=2, client=client)

    with pytest.raises(OpenAITranslatorCountError):
        translator.translate_batch(
            ["Hello", "World"],
            source_language="English",
            target_language="French",
        )


def test_translate_batch_uses_supported_responses_api_kwargs() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"translations": [{"id": 1, "translation": "Bonjour"}]}'
    )

    translator = OpenAITranslator(api_key="test-key", model="gpt-4.1-mini", timeout=42.0, client=client)

    translator.translate_batch(
        ["Hello"],
        source_language="English",
        target_language="French",
    )

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4.1-mini"
    assert kwargs["input"] == (
        "Translate the following texts from English to French.\n"
        "Return only a JSON object with a translations array.\n"
        "Use exactly the same ids provided in the input JSON.\n"
        "Keep placeholders, markup, and code-like tokens unchanged.\n"
        "Use deterministic, concise, natural phrasing.\n\n"
        "Inputs JSON:\n"
        "[\n"
        "  {\n"
        "    \"id\": 1,\n"
        "    \"text\": \"Hello\"\n"
        "  }\n"
        "]"
    )
    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert kwargs["text"]["format"]["name"] == "translation_batch"
    assert kwargs["text"]["format"]["strict"] is True
    assert kwargs["text"]["format"]["schema"]["type"] == "object"
    assert kwargs["timeout"] == 42.0
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_translate_batch_raises_for_unrecoverable_request_error() -> None:
    client = Mock()
    client.responses.create.side_effect = RuntimeError("boom")

    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-5.5",
        max_retries=1,
        retry_delay=0,
        client=client,
    )

    with pytest.raises(OpenAITranslatorRequestError):
        translator.translate_batch(
            ["Hello"],
            source_language="English",
            target_language="French",
        )


def test_translate_batch_keeps_requested_order_when_response_is_reordered() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"translations": [{"id": 2, "translation": "Salut"}, {"id": 1, "translation": "Bonjour"}]}'
    )

    translator = OpenAITranslator(api_key="test-key", model="gpt-5.5", batch_size=2, client=client)

    result = translator.translate_batch(
        ["Hello", "World"],
        source_language="English",
        target_language="French",
    )

    assert result == ["Bonjour", "Salut"]


def test_translate_batch_allows_multiline_translations() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "translations": [
                    {"id": 1, "translation": "Bonjour\nle monde"},
                    {"id": 2, "translation": "Salut"},
                ]
            }
        )
    )

    translator = OpenAITranslator(api_key="test-key", model="gpt-5.5", batch_size=2, client=client)

    result = translator.translate_batch(
        ["Hello world", "Hi"],
        source_language="English",
        target_language="French",
    )

    assert result == ["Bonjour\nle monde", "Salut"]


def test_translate_batch_raises_on_missing_ids() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"translations": [{"id": 1, "translation": "Bonjour"}]}'
    )

    translator = OpenAITranslator(api_key="test-key", model="gpt-5.5", batch_size=2, max_retries=1, client=client)

    with pytest.raises(OpenAITranslatorCountError, match="Missing translation ids"):
        translator.translate_batch(
            ["Hello", "World"],
            source_language="English",
            target_language="French",
        )


def test_translate_batch_raises_on_duplicated_ids() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"translations": [{"id": 1, "translation": "Bonjour"}, {"id": 1, "translation": "Salut"}]}'
    )

    translator = OpenAITranslator(api_key="test-key", model="gpt-5.5", batch_size=2, max_retries=1, client=client)

    with pytest.raises(OpenAITranslatorCountError, match="Duplicate translation ids"):
        translator.translate_batch(
            ["Hello", "World"],
            source_language="English",
            target_language="French",
        )


def test_translate_batch_retries_on_invalid_json() -> None:
    client = Mock()
    client.responses.create.side_effect = [
        SimpleNamespace(output_text="{not-json"),
        SimpleNamespace(
            output_text='{"translations": [{"id": 1, "translation": "Bonjour"}, {"id": 2, "translation": "Salut"}]}'
        ),
    ]

    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-5.5",
        batch_size=2,
        max_retries=2,
        retry_delay=0,
        client=client,
    )

    result = translator.translate_batch(
        ["Hello", "World"],
        source_language="English",
        target_language="French",
    )

    assert result == ["Bonjour", "Salut"]
    assert client.responses.create.call_count == 2


def test_translate_batch_raises_on_extra_ids() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=(
            '{"translations": ['
            '{"id": 1, "translation": "Bonjour"}, '
            '{"id": 2, "translation": "Salut"}, '
            '{"id": 3, "translation": "Extra"}]}'
        )
    )

    translator = OpenAITranslator(api_key="test-key", model="gpt-5.5", batch_size=2, max_retries=1, client=client)

    with pytest.raises(OpenAITranslatorCountError, match="Unexpected translation ids"):
        translator.translate_batch(
            ["Hello", "World"],
            source_language="English",
            target_language="French",
        )
