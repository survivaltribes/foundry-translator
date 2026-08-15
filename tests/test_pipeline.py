from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import replace
from dataclasses import dataclass

from foundry_translator.pipeline import (
    Pipeline,
    TranslationProgressReporter,
    format_duration,
    format_progress_bar,
    format_translation_speed,
)
from foundry_translator.translator import DummyTranslator
from foundry_translator.translator import Translator
from foundry_translator.scanner import TranslationEntry


@dataclass(slots=True)
class FakeFailedEntry:
    file: Path
    json_path: str
    reason: str
    missing_placeholders: list[str]
    unexpected_placeholders: list[str]
    debug_dir: Path | None = None
    field: str | None = None


class ResumableTestTranslator(Translator):
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        super().__init__()
        self.batch_size = 1
        self.fail_on_call = fail_on_call
        self.call_count = 0
        self.seen_sources: list[str] = []

    def translate(self, entries: list[TranslationEntry]) -> list[TranslationEntry]:
        self.call_count += 1
        if self.fail_on_call is not None and self.call_count == self.fail_on_call:
            raise RuntimeError("forced interruption")

        translated: list[TranslationEntry] = []
        for entry in entries:
            self.seen_sources.append(entry.source)
            translated.append(replace(entry, source=f"T:{entry.source}"))
        return translated


class PartialFailureTranslator(Translator):
    def __init__(self) -> None:
        super().__init__()
        self.failed_entries: list[FakeFailedEntry] = []

    def translate(self, entries: list[TranslationEntry]) -> list[TranslationEntry]:
        self.failed_entries = [
            FakeFailedEntry(
                file=entries[0].file,
                json_path="$['description']",
                reason="Placeholder mismatch after translation: missing=['__FT_HTML_00001__'] unexpected=[]",
                missing_placeholders=["__FT_HTML_00001__"],
                unexpected_placeholders=[],
                field=entries[0].field,
            )
        ]
        return [replace(entries[1], source=f"T:{entries[1].source}")]


def test_translation_progress_reporter_emits_batch_and_summary_output(capsys) -> None:
    reporter = TranslationProgressReporter(total_texts=3, total_files=1, total_unique_texts=3)

    reporter.on_request_completed(
        batch_number=1,
        total_batches=2,
        batch_size=2,
        prompt_size=256,
        elapsed_seconds=0.4,
        current_file="compendium.json",
        translated_count=2,
        total_texts=3,
    )

    captured = capsys.readouterr()
    assert "batch 1/2" in captured.out
    assert "2/3" in captured.out
    assert "elapsed" in captured.out
    assert "ETA" in captured.out
    assert "finish" in captured.out
    assert "Translation summary:" in reporter.summary()


def test_format_duration_formats_hours_minutes_and_days() -> None:
    assert format_duration(581.4) == "9m 41s"
    assert format_duration(3900) == "1h 5m"
    assert format_duration(90000) == "1d 1h"


def test_format_progress_bar_formats_expected_percentages() -> None:
    assert format_progress_bar(0.0) == "[░░░░░░░░░░░░░░░░░░░░] 0.0%"
    assert format_progress_bar(0.25) == "[█████░░░░░░░░░░░░░░░] 25.0%"
    assert format_progress_bar(0.5) == "[██████████░░░░░░░░░░] 50.0%"
    assert format_progress_bar(1.0) == "[████████████████████] 100.0%"


def test_format_progress_bar_respects_width_and_fallback(monkeypatch) -> None:
    bar = format_progress_bar(0.5, width=8)
    assert bar.startswith("[") and bar.endswith("] 50.0%")
    assert len(bar.split("]", 1)[0]) == 9

    class DummyStdout:
        encoding = "ascii"

    monkeypatch.setattr(sys, "stdout", DummyStdout())
    assert format_progress_bar(0.5, width=8) == "[####----] 50.0%"


def test_format_translation_speed_handles_calculating_and_pluralization() -> None:
    assert format_translation_speed(0, 0.5) == "Speed: calculating..."
    assert format_translation_speed(0, 10.0) == "Speed: 0 texts/min"
    assert format_translation_speed(34, 60.0) == "Speed: 34 texts/min"
    assert format_translation_speed(1, 60.0) == "Speed: 1 text/min"
    assert format_translation_speed(1, 120.0) == "Speed: 0.01 texts/s"
    assert format_translation_speed(3, 200.0) == "Speed: 0.01 texts/s"


def test_pipeline_writes_translated_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)

    source_file = input_dir / "compendium.json"
    source_file.write_text(json.dumps({"name": "Hero", "ignored": "keep me"}), encoding="utf-8")

    result = Pipeline(translator=DummyTranslator()).run(input_dir, output_dir)

    assert result.scanned_files == 1
    assert result.translated_entries == 1
    assert result.errors == 0
    assert (output_dir / "compendium.json").exists()
    assert (output_dir / "compendium.translated.json").exists()

    written = json.loads((output_dir / "compendium.translated.json").read_text(encoding="utf-8"))
    assert written["name"] == "Hero"
    assert written["ignored"] == "keep me"


def test_pipeline_run_only_file_filters_documents_and_entries(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)

    first_file = input_dir / "first.json"
    second_file = input_dir / "second.json"
    first_file.write_text(json.dumps({"name": "Hero"}), encoding="utf-8")
    second_file.write_text(json.dumps({"name": "Villain"}), encoding="utf-8")

    result = Pipeline(translator=DummyTranslator()).run(
        input_dir,
        output_dir,
        only_file="second.json",
    )

    assert result.scanned_files == 1
    assert result.translated_entries == 1
    assert (output_dir / "second.json").exists()
    assert (output_dir / "second.translated.json").exists()
    assert not (output_dir / "first.translated.json").exists()


def test_pipeline_run_limit_truncates_translation_entries(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)

    first_file = input_dir / "first.json"
    second_file = input_dir / "second.json"
    first_file.write_text(json.dumps({"name": "Hero", "description": "Brave"}), encoding="utf-8")
    second_file.write_text(json.dumps({"name": "Villain"}), encoding="utf-8")

    result = Pipeline(translator=DummyTranslator()).run(
        input_dir,
        output_dir,
        limit=1,
    )

    assert result.translated_entries == 1
    assert (output_dir / "first.translated.json").exists() or (output_dir / "second.translated.json").exists()


def test_pipeline_counts_and_reports_failed_entries_without_stopping(tmp_path: Path, capsys) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)

    source_file = input_dir / "actors.json"
    source_file.write_text(
        json.dumps({"description": "Hello", "feature": "Brave", "name": "Hero"}),
        encoding="utf-8",
    )

    result = Pipeline(translator=PartialFailureTranslator()).run(input_dir, output_dir)

    assert result.scanned_files == 1
    assert result.translated_entries == 1
    assert result.errors == 1
    assert (output_dir / "actors.translated.json").exists()

    captured = capsys.readouterr()
    assert "Failed entries:" in captured.out
    assert "actors.json.description" in captured.out
    assert "$['description']" in captured.out
    assert "missing: ['__FT_HTML_00001__']" in captured.out
    assert "unexpected: []" in captured.out


def test_pipeline_resume_interrupted_run_persists_progress(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)

    source_file = input_dir / "compendium.json"
    source_file.write_text(
        json.dumps(
            {
                "name": "Hero",
                "description": "Brave",
            }
        ),
        encoding="utf-8",
    )

    translator = ResumableTestTranslator(fail_on_call=2)
    pipeline = Pipeline(translator=translator)

    try:
        pipeline.run(input_dir, output_dir, resume=True)
    except RuntimeError as exc:
        assert "forced interruption" in str(exc)
    else:  # pragma: no cover - guard
        raise AssertionError("Expected forced interruption")

    progress_path = output_dir / "progress.json"
    assert progress_path.exists()
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["completed_chunk_indices"] == [0]
    assert payload["translated_entry_count"] == 1
    assert payload["total_entry_count"] == 2


def test_pipeline_resume_continues_after_interrupted_run(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)

    source_file = input_dir / "compendium.json"
    source_file.write_text(
        json.dumps(
            {
                "name": "Hero",
                "description": "Brave",
            }
        ),
        encoding="utf-8",
    )

    first_translator = ResumableTestTranslator(fail_on_call=2)
    first_pipeline = Pipeline(translator=first_translator)
    try:
        first_pipeline.run(input_dir, output_dir, resume=True)
    except RuntimeError:
        pass

    resumed_translator = ResumableTestTranslator()
    resumed_pipeline = Pipeline(translator=resumed_translator)
    result = resumed_pipeline.run(input_dir, output_dir, resume=True)

    assert result.translated_entries == 2
    assert resumed_translator.call_count == 1
    assert resumed_translator.seen_sources == ["Brave"]

    translated_payload = json.loads((output_dir / "compendium.translated.json").read_text(encoding="utf-8"))
    assert translated_payload["name"] == "T:Hero"
    assert translated_payload["description"] == "T:Brave"


def test_pipeline_resume_completed_run_has_full_progress(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)

    source_file = input_dir / "compendium.json"
    source_file.write_text(
        json.dumps(
            {
                "name": "Hero",
                "description": "Brave",
            }
        ),
        encoding="utf-8",
    )

    translator = ResumableTestTranslator()
    pipeline = Pipeline(translator=translator)
    result = pipeline.run(input_dir, output_dir, resume=True)

    assert result.translated_entries == 2
    progress_path = output_dir / "progress.json"
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["completed_chunk_indices"] == [0, 1]
    assert payload["translated_entry_count"] == 2
    assert payload["total_entry_count"] == 2


def test_pipeline_resume_rejects_corrupted_progress_file(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_file = input_dir / "compendium.json"
    source_file.write_text(json.dumps({"name": "Hero"}), encoding="utf-8")
    (output_dir / "progress.json").write_text("{not-json", encoding="utf-8")

    translator = ResumableTestTranslator()
    pipeline = Pipeline(translator=translator)

    try:
        pipeline.run(input_dir, output_dir, resume=True)
    except ValueError as exc:
        assert "Corrupted progress file" in str(exc)
    else:  # pragma: no cover - guard
        raise AssertionError("Expected corrupted progress error")


def test_pipeline_resume_rejects_changed_input_directory(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)

    source_file = input_dir / "compendium.json"
    source_file.write_text(json.dumps({"name": "Hero", "description": "Brave"}), encoding="utf-8")

    first_translator = ResumableTestTranslator(fail_on_call=2)
    first_pipeline = Pipeline(translator=first_translator)
    try:
        first_pipeline.run(input_dir, output_dir, resume=True)
    except RuntimeError:
        pass

    source_file.write_text(json.dumps({"name": "Hero", "description": "Changed"}), encoding="utf-8")

    resumed_translator = ResumableTestTranslator()
    resumed_pipeline = Pipeline(translator=resumed_translator)

    try:
        resumed_pipeline.run(input_dir, output_dir, resume=True)
    except ValueError as exc:
        assert "Cannot resume: input files changed" in str(exc)
    else:  # pragma: no cover - guard
        raise AssertionError("Expected changed input refusal")
