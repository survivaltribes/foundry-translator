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


def test_build_parser_supports_replay_restore_and_fast_run_options() -> None:
    parser = build_parser()

    run_args = parser.parse_args([
        "run",
        "--input",
        "input",
        "--output",
        "output",
        "--only-file",
        "sample.json",
        "--limit",
        "5",
    ])
    assert run_args.command == "run"
    assert run_args.only_file == "sample.json"
    assert run_args.limit == 5
    assert run_args.resume is False

    resumed_run_args = parser.parse_args([
        "run",
        "--input",
        "input",
        "--output",
        "output",
        "--resume",
    ])
    assert resumed_run_args.command == "run"
    assert resumed_run_args.resume is True

    replay_args = parser.parse_args([
        "replay-restore",
        "--debug-dir",
        "debug/restore_duplicate_placeholders_xxxxx",
    ])
    assert replay_args.command == "replay-restore"
    assert replay_args.debug_dir == "debug/restore_duplicate_placeholders_xxxxx"

    resume_args = parser.parse_args([
        "replay-restore",
        "--resume-from-debug",
        "debug/restore_duplicate_placeholders_xxxxx",
    ])
    assert resume_args.command == "replay-restore"
    assert resume_args.debug_dir == "debug/restore_duplicate_placeholders_xxxxx"


def test_build_translator_uses_openai_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    translator = build_translator()

    assert isinstance(translator, OpenAITranslator)


def test_build_translator_honors_injected_translator() -> None:
    translator = DummyTranslator()

    assert build_translator(translator) is translator
