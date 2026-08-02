"""End-to-end translation pipeline for Babele JSON exports."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from .scanner import Scanner
from .translation_cache import TranslationCache
from .translator import Translator
from .writer import JsonWriter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineResult:
    """Summary of a completed translation run."""

    input_dir: Path
    output_dir: Path
    scanned_files: int
    translated_entries: int
    errors: int
    duration_seconds: float


class Pipeline:
    """Orchestrate scanning, translation, and writing for a whole folder."""

    def __init__(
        self,
        *,
        translator: Translator,
        writer: JsonWriter | None = None,
        cache: TranslationCache | None = None,
    ) -> None:
        self.translator = translator
        self.writer = writer or JsonWriter()
        self.cache = cache
        if self.cache is not None and hasattr(self.translator, "cache"):
            self.translator.cache = self.cache

    def run(self, input_dir: Path | str, output_dir: Path | str) -> PipelineResult:
        input_path = Path(input_dir).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()

        started_at = perf_counter()
        output_path.mkdir(parents=True, exist_ok=True)

        scanner = Scanner(input_path)
        documents, entries, issues = scanner.scan()

        translated_entries = self.translator.translate(entries)

        for document in documents:
            destination = output_path / document.path.relative_to(input_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(document.path.read_bytes())

        for source_file in {entry.file for entry in translated_entries}:
            relative_source = source_file.relative_to(input_path)
            translated_file = self.writer.write(source_file, translated_entries)
            destination = output_path / relative_source.with_suffix(".translated.json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(translated_file.read_bytes())

        duration = perf_counter() - started_at

        result = PipelineResult(
            input_dir=input_path,
            output_dir=output_path,
            scanned_files=len(documents),
            translated_entries=len(translated_entries),
            errors=len(issues),
            duration_seconds=duration,
        )

        logger.info(
            "Pipeline completed: files=%s translated=%s errors=%s duration=%.2fs",
            result.scanned_files,
            result.translated_entries,
            result.errors,
            result.duration_seconds,
        )
        return result
