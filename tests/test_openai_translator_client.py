from __future__ import annotations

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
    client.responses.create.return_value = SimpleNamespace(output_text="Bonjour\nSalut")

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
        SimpleNamespace(output_text="Bonjour"),
        SimpleNamespace(output_text="Bonjour\nSalut"),
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
    client.responses.create.return_value = SimpleNamespace(output_text="Bonjour")

    translator = OpenAITranslator(api_key="test-key", model="gpt-5.5", batch_size=2, client=client)

    with pytest.raises(OpenAITranslatorCountError):
        translator.translate_batch(
            ["Hello", "World"],
            source_language="English",
            target_language="French",
        )


def test_translate_batch_uses_supported_responses_api_kwargs() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(output_text="Bonjour")

    translator = OpenAITranslator(api_key="test-key", model="gpt-4.1-mini", timeout=42.0, client=client)

    translator.translate_batch(
        ["Hello"],
        source_language="English",
        target_language="French",
    )

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4.1-mini"
    assert kwargs["input"] == "Translate the following texts from English to French.\nReturn exactly one translated line per input line, preserving the same order.\nDo not add commentary, numbering, bullets, or extra prose.\nKeep placeholders, markup, and code-like tokens unchanged.\nUse deterministic, concise, natural phrasing.\n\nInputs:\n1. Hello"
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
