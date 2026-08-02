from __future__ import annotations

import json
from pathlib import Path

from foundry_translator.pipeline import Pipeline, TranslationProgressReporter
from foundry_translator.translator import DummyTranslator


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
    assert "batch=1/2" in captured.out
    assert "translated=2/3" in captured.out
    assert "file=compendium.json" in captured.out
    assert "Translation summary:" in reporter.summary()


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
