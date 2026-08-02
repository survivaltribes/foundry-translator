from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from foundry_translator.openai_translator import OpenAITranslator
from foundry_translator.pipeline import Pipeline
from foundry_translator.protect import Protect
from foundry_translator.translator import DummyTranslator
from foundry_translator.writer import JsonWriter

UUID_PATTERN = re.compile(r"@UUID\[[^\]]+\]")
EMBED_PATTERN = re.compile(r"@Embed\[[^\]]+\]")
ROLL_PATTERN = re.compile(r"\[\[[^\]]+\]\]")
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^\)]+\)")
HTML_TAG_PATTERN = re.compile(r"</?([a-zA-Z0-9]+)")
PLACEHOLDER_PATTERN = re.compile(r"__FT_[A-Z_]+_\d{5}__|__FOUNDRY_PLACEHOLDER_[0-9]+__")


@dataclass(slots=True)
class ValidationFailure:
    file: Path
    path: str
    expected: Any
    actual: Any

    def format(self) -> str:
        return f"{self.file}: {self.path} expected={self.expected!r} actual={self.actual!r}"


@dataclass(slots=True)
class ValidationReport:
    files_processed: int = 0
    translation_entries: int = 0
    elapsed_seconds: float = 0.0
    placeholders_protected: int = 0
    placeholders_restored: int = 0
    failures: list[ValidationFailure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            "Acceptance validation report",
            f"- files processed: {self.files_processed}",
            f"- translation entries: {self.translation_entries}",
            f"- elapsed time: {self.elapsed_seconds:.3f}s",
            f"- placeholders protected: {self.placeholders_protected}",
            f"- placeholders restored: {self.placeholders_restored}",
            f"- validation failures: {len(self.failures)}",
            f"- warnings: {len(self.warnings)}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        if self.failures:
            lines.append("Failures:")
            lines.extend(f"  - {failure.format()}" for failure in self.failures)
        return "\n".join(lines)


def _resolve_acceptance_source_dir() -> Path:
    explicit = os.getenv("FOUNDRY_TRANSLATOR_ACCEPTANCE_SOURCE")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path("/mnt/c/Users/survi/AppData/Local/FoundryVTT/Data/modules/dnd-heroes-borderlands/fr/compendium-export"),
            Path("C:/Users/survi/AppData/Local/FoundryVTT/Data/modules/dnd-heroes-borderlands/fr/compendium-export"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pytest.skip(
        "Acceptance test data not available. Set FOUNDRY_TRANSLATOR_ACCEPTANCE_SOURCE to the exported compendium directory."
    )


def _build_translator() -> Any:
    use_openai = os.getenv("FOUNDRY_TRANSLATOR_USE_OPENAI", "").lower() in {"1", "true", "yes"}
    use_openai = use_openai or any(argument == "--acceptance-openai" for argument in sys.argv)

    if not use_openai:
        return DummyTranslator()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when FOUNDRY_TRANSLATOR_USE_OPENAI=1")

    return OpenAITranslator(
        api_key=api_key,
        model="gpt-4.1-mini",
        target_language="French",
        batch_size=5,
    )


def _compare_structure(expected: Any, actual: Any, path: str, failures: list[ValidationFailure], file: Path) -> None:
    if type(expected) is not type(actual):
        failures.append(ValidationFailure(file=file, path=path, expected=type(expected).__name__, actual=type(actual).__name__))
        return

    if isinstance(expected, dict):
        if set(expected) != set(actual):
            failures.append(
                ValidationFailure(
                    file=file,
                    path=path,
                    expected=sorted(expected.keys()),
                    actual=sorted(actual.keys()),
                )
            )
            return
        for key in expected:
            _compare_structure(expected[key], actual[key], f"{path}.{key}" if path else str(key), failures, file)
        return

    if isinstance(expected, list):
        if len(expected) != len(actual):
            failures.append(ValidationFailure(file=file, path=path, expected=len(expected), actual=len(actual)))
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare_structure(left, right, f"{path}[{index}]" if path else f"[{index}]", failures, file)
        return

    if isinstance(expected, str):
        return

    if expected != actual:
        failures.append(ValidationFailure(file=file, path=path, expected=expected, actual=actual))


def _assert_string_invariants(source_value: str, translated_value: str, file: Path, path: str, failures: list[ValidationFailure]) -> None:
    if UUID_PATTERN.findall(source_value) != UUID_PATTERN.findall(translated_value):
        failures.append(ValidationFailure(file=file, path=path, expected=UUID_PATTERN.findall(source_value), actual=UUID_PATTERN.findall(translated_value)))
    if EMBED_PATTERN.findall(source_value) != EMBED_PATTERN.findall(translated_value):
        failures.append(ValidationFailure(file=file, path=path, expected=EMBED_PATTERN.findall(source_value), actual=EMBED_PATTERN.findall(translated_value)))
    if ROLL_PATTERN.findall(source_value) != ROLL_PATTERN.findall(translated_value):
        failures.append(ValidationFailure(file=file, path=path, expected=ROLL_PATTERN.findall(source_value), actual=ROLL_PATTERN.findall(translated_value)))
    if IMAGE_PATTERN.findall(source_value) != IMAGE_PATTERN.findall(translated_value):
        failures.append(ValidationFailure(file=file, path=path, expected=IMAGE_PATTERN.findall(source_value), actual=IMAGE_PATTERN.findall(translated_value)))

    if "<" in source_value or "<" in translated_value:
        source_tags = HTML_TAG_PATTERN.findall(source_value)
        translated_tags = HTML_TAG_PATTERN.findall(translated_value)
        if source_tags != translated_tags:
            failures.append(ValidationFailure(file=file, path=path, expected=source_tags, actual=translated_tags))

    if PLACEHOLDER_PATTERN.search(translated_value):
        failures.append(ValidationFailure(file=file, path=path, expected="no unresolved placeholders", actual=translated_value))


def _walk_strings(expected: Any, actual: Any, path: str, file: Path, failures: list[ValidationFailure]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in expected:
            _walk_strings(expected[key], actual[key], f"{path}.{key}" if path else str(key), file, failures)
        return

    if isinstance(expected, list) and isinstance(actual, list):
        for index, (left, right) in enumerate(zip(expected, actual)):
            _walk_strings(left, right, f"{path}[{index}]" if path else f"[{index}]", file, failures)
        return

    if isinstance(expected, str) and isinstance(actual, str):
        _assert_string_invariants(expected, actual, file, path, failures)


def test_acceptance_real_foundry_compendium(tmp_path: Path) -> None:
    source_dir = _resolve_acceptance_source_dir()
    working_dir = tmp_path / "compendium-export"
    output_dir = tmp_path / "output"

    shutil.copytree(source_dir, working_dir)

    initial_files = sorted(path for path in working_dir.rglob("*.json") if path.is_file())

    translator = _build_translator()
    writer = JsonWriter()
    pipeline = Pipeline(translator=translator, writer=writer)

    report = ValidationReport(files_processed=len(initial_files))
    report.translation_entries = 0

    protector = Protect()
    protected_count = 0
    for json_file in initial_files:
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.failures.append(ValidationFailure(file=json_file, path="<root>", expected="valid JSON", actual=str(exc)))
            continue

        def collect_strings(value: Any, path: str = "") -> None:
            nonlocal protected_count
            if isinstance(value, dict):
                for key, nested in value.items():
                    collect_strings(nested, f"{path}.{key}" if path else str(key))
            elif isinstance(value, list):
                for index, nested in enumerate(value):
                    collect_strings(nested, f"{path}[{index}]" if path else f"[{index}]")
            elif isinstance(value, str):
                protected = protector.protect(value)
                protected_count += len(protected.placeholders)

        collect_strings(payload)

    report.placeholders_protected = protected_count

    start = time.perf_counter()
    result = pipeline.run(working_dir, output_dir)
    report.elapsed_seconds = time.perf_counter() - start
    report.translation_entries = result.translated_entries

    translated_outputs = sorted(path for path in output_dir.rglob("*.translated.json") if path.is_file())
    if len(translated_outputs) != len(initial_files):
        report.failures.append(ValidationFailure(file=output_dir, path="<root>", expected=len(initial_files), actual=len(translated_outputs)))

    for source_file in initial_files:
        relative_path = source_file.relative_to(working_dir)
        translated_output = output_dir / relative_path.with_suffix(".translated.json")
        if not translated_output.exists():
            report.failures.append(ValidationFailure(file=source_file, path="<output>", expected="translated file", actual="missing"))
            continue

        try:
            source_payload = json.loads(source_file.read_text(encoding="utf-8"))
            translated_payload = json.loads(translated_output.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.failures.append(ValidationFailure(file=translated_output, path="<root>", expected="valid JSON", actual=str(exc)))
            continue

        _compare_structure(source_payload, translated_payload, "<root>", report.failures, source_file)
        _walk_strings(source_payload, translated_payload, "<root>", source_file, report.failures)

        placeholder_count = sum(1 for value in re.finditer(PLACEHOLDER_PATTERN.pattern, json.dumps(translated_payload)))
        report.placeholders_restored = max(0, protected_count - placeholder_count)

        try:
            writer.load(translated_output)
        except Exception as exc:  # pragma: no cover - defensive safety branch
            report.failures.append(ValidationFailure(file=translated_output, path="<root>", expected="loadable JSON", actual=str(exc)))

    if not report.failures:
        report.warnings.append("No validation issues detected")

    print(report.to_text())
    assert not report.failures, report.to_text()
