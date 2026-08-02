from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .scanner import Scanner
from .translation_cache import TranslationCache
from .translator import Translator
from .writer import JsonWriter

logger = logging.getLogger(__name__)


class TranslationProgressReporter:
    """Report translation progress and summary details to the CLI."""

    def __init__(self, *, total_texts: int, total_files: int, total_unique_texts: int) -> None:
        self.total_texts = total_texts
        self.total_files = total_files
        self.total_unique_texts = total_unique_texts
        self.started_at = perf_counter()
        self.request_count = 0
        self.batch_sizes: list[int] = []
        self.prompt_sizes: list[int] = []
        self.request_durations: list[float] = []
        self.completed_batches = 0
        self.completed_texts = 0
        self.current_file = "n/a"

    def attach(self, translator: Any) -> None:
        if hasattr(translator, "_progress_reporter"):
            translator._progress_reporter = self

    def on_request_completed(
        self,
        *,
        batch_number: int,
        total_batches: int,
        batch_size: int,
        prompt_size: int,
        elapsed_seconds: float,
        current_file: str | None,
        translated_count: int,
        total_texts: int,
        failed: bool = False,
    ) -> None:
        self.request_count += 1
        self.batch_sizes.append(batch_size)
        self.prompt_sizes.append(prompt_size)
        self.request_durations.append(elapsed_seconds)
        self.completed_batches = batch_number
        self.completed_texts = translated_count
        self.current_file = current_file or self.current_file
        if not failed:
            self.completed_texts = translated_count
        percentage = 100.0 if total_texts == 0 else (translated_count / total_texts) * 100.0
        elapsed = perf_counter() - self.started_at
        remaining_batches = max(0, total_batches - batch_number)
        if self.request_durations:
            avg_duration = sum(self.request_durations) / len(self.request_durations)
            eta_seconds = avg_duration * remaining_batches
        else:
            eta_seconds = 0.0
        eta_text = f"eta={eta_seconds:>6.1f}s" if remaining_batches else "eta=0.0s"
        print(
            f"batch={batch_number}/{total_batches} translated={translated_count}/{total_texts} "
            f"{percentage:>5.1f}% file={self.current_file} prompt={prompt_size} chars"
            f"elapsed={elapsed:>6.1f}s {eta_text}",
            flush=True,
        )

    def summary(self) -> str:
        average_batch_size = (sum(self.batch_sizes) / len(self.batch_sizes)) if self.batch_sizes else 0.0
        average_prompt_size = (sum(self.prompt_sizes) / len(self.prompt_sizes)) if self.prompt_sizes else 0.0
        average_request_duration = (sum(self.request_durations) / len(self.request_durations)) if self.request_durations else 0.0
        total_elapsed = perf_counter() - self.started_at
        return (
            "Translation summary: "
            f"files_processed={self.total_files} unique_texts={self.total_unique_texts} "
            f"translated_texts={self.completed_texts} openai_requests={self.request_count} "
            f"average_batch_size={average_batch_size:.2f} average_prompt_size={average_prompt_size:.2f} "
            f"average_request_duration={average_request_duration:.3f}s total_elapsed={total_elapsed:.3f}s"
        )


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

        reporter = TranslationProgressReporter(
            total_texts=len(entries),
            total_files=len(documents),
            total_unique_texts=len({entry.source for entry in entries}),
        )
        reporter.attach(self.translator)

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
        print(reporter.summary(), flush=True)
        return result
