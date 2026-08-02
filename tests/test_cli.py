from __future__ import annotations

import os

import pytest

from foundry_translator.cli import build_parser, build_translator
from foundry_translator.openai_translator import OpenAITranslator
from foundry_translator.translator import DummyTranslator


def test_build_parser_requires_a_subcommand() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([])

    assert excinfo.value.code == 2


def test_build_translator_uses_openai_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    translator = build_translator()

    assert isinstance(translator, OpenAITranslator)


def test_build_translator_honors_injected_translator() -> None:
    translator = DummyTranslator()

    assert build_translator(translator) is translator
