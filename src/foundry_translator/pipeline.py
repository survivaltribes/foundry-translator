from __future__ import annotations

import logging
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .scanner import Scanner
from .translation_cache import TranslationCache
from .translator import Translator
from .writer import JsonWriter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResumeProgress:
    """Serializable resume progress persisted between resumable runs."""

    input_signature: str
    completed_chunk_indices: list[int]
    current_document: str
    translated_entry_count: int
    total_entry_count: int
    elapsed_time_seconds: float
    translated_entries: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_signature": self.input_signature,
            "completed_chunk_indices": self.completed_chunk_indices,
            "current_document": self.current_document,
            "translated_entry_count": self.translated_entry_count,
            "total_entry_count": self.total_entry_count,
            "elapsed_time_seconds": self.elapsed_time_seconds,
            "translated_entries": self.translated_entries,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResumeProgress":
        required = {
            "input_signature",
            "completed_chunk_indices",
            "current_document",
            "translated_entry_count",
            "total_entry_count",
            "elapsed_time_seconds",
            "translated_entries",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Corrupted progress file: missing keys {missing}")

        completed = payload["completed_chunk_indices"]
        translated_entries = payload["translated_entries"]
        if not isinstance(completed, list) or not all(isinstance(item, int) for item in completed):
            raise ValueError("Corrupted progress file: completed_chunk_indices must be a list of integers")
        if not isinstance(translated_entries, list) or not all(isinstance(item, dict) for item in translated_entries):
            raise ValueError("Corrupted progress file: translated_entries must be a list of objects")

        try:
            translated_entry_count = int(payload["translated_entry_count"])
            total_entry_count = int(payload["total_entry_count"])
            elapsed_time_seconds = float(payload["elapsed_time_seconds"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Corrupted progress file: invalid numeric fields") from exc

        return cls(
            input_signature=str(payload["input_signature"]),
            completed_chunk_indices=sorted(set(completed)),
            current_document=str(payload["current_document"]),
            translated_entry_count=translated_entry_count,
            total_entry_count=total_entry_count,
            elapsed_time_seconds=elapsed_time_seconds,
            translated_entries=translated_entries,
        )


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

    def run(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        *,
        only_file: str | None = None,
        limit: int | None = None,
        resume: bool = False,
    ) -> PipelineResult:
        input_path = Path(input_dir).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()

        return self.run_filtered(input_path, output_path, only_file=only_file, limit=limit, resume=resume)

    def run_filtered(
        self,
        input_path: Path,
        output_path: Path,
        *,
        only_file: str | None = None,
        limit: int | None = None,
        resume: bool = False,
    ) -> PipelineResult:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        started_at = perf_counter()
        output_path.mkdir(parents=True, exist_ok=True)

        scanner = Scanner(input_path)
        documents, entries, issues = scanner.scan()

        if only_file is not None:
            documents, entries = self._filter_single_document(input_path, documents, entries, only_file)

        if limit is not None:
            entries = entries[:limit]

        reporter = TranslationProgressReporter(
            total_texts=len(entries),
            total_files=len(documents),
            total_unique_texts=len({entry.source for entry in entries}),
        )
        reporter.attach(self.translator)

        if resume:
            translated_entries = self._run_resumable_translation(
                input_path=input_path,
                output_path=output_path,
                entries=entries,
                only_file=only_file,
                limit=limit,
                started_at=started_at,
            )
        else:
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

    def _run_resumable_translation(
        self,
        *,
        input_path: Path,
        output_path: Path,
        entries: list[Any],
        only_file: str | None,
        limit: int | None,
        started_at: float,
    ) -> list[Any]:
        progress_path = output_path / "progress.json"
        input_signature = self._build_input_signature(entries=entries, only_file=only_file, limit=limit)

        chunk_size = max(1, int(getattr(self.translator, "batch_size", len(entries) or 1)))
        entry_chunks = [entries[index : index + chunk_size] for index in range(0, len(entries), chunk_size)]

        progress = self._load_progress(progress_path)
        if progress is None:
            logger.info("resume diagnostics: no progress file found; starting fresh")
            progress = ResumeProgress(
                input_signature=input_signature,
                completed_chunk_indices=[],
                current_document="n/a",
                translated_entry_count=0,
                total_entry_count=len(entries),
                elapsed_time_seconds=0.0,
                translated_entries=[],
            )
        else:
            logger.info(
                "resume diagnostics: loaded progress completed_chunks=%s translated_entries=%s total_entries=%s current_document=%s",
                sorted(progress.completed_chunk_indices),
                progress.translated_entry_count,
                progress.total_entry_count,
                progress.current_document,
            )
            if progress.input_signature != input_signature:
                raise ValueError("Cannot resume: input files changed since previous run")

        translated_by_key: dict[tuple[str, tuple[Any, ...], str], Any] = {}
        for payload in progress.translated_entries:
            restored = self._entry_from_progress(payload)
            key = self._entry_key(restored)
            translated_by_key[key] = restored

        completed_chunks = set(progress.completed_chunk_indices)

        if hasattr(self.translator, "__dict__"):
            self.translator.__dict__["_resume_diagnostics_active"] = True

        for chunk_index, chunk_entries in enumerate(entry_chunks):
            if chunk_index in completed_chunks:
                logger.info(
                    "resume diagnostics: chunk=%s source=loaded_state entry_count=%s",
                    chunk_index,
                    len(chunk_entries),
                )
                continue

            logger.info(
                "resume diagnostics: transition loaded_state->fresh_translation chunk=%s loaded_entries=%s pending_entries=%s current_document=%s",
                chunk_index,
                len(translated_by_key),
                len(chunk_entries),
                str(chunk_entries[0].file) if chunk_entries else "n/a",
            )

            if hasattr(self.translator, "__dict__"):
                self.translator.__dict__["_resume_diagnostics_chunk_index"] = chunk_index
                self.translator.__dict__["_resume_diagnostics_loaded_entries"] = len(translated_by_key)

            translated_chunk_entries = self.translator.translate(chunk_entries)
            logger.info(
                "resume diagnostics: chunk=%s source=fresh_translation produced_entries=%s",
                chunk_index,
                len(translated_chunk_entries),
            )
            for translated_entry in translated_chunk_entries:
                translated_by_key[self._entry_key(translated_entry)] = translated_entry

            completed_chunks.add(chunk_index)
            progress.completed_chunk_indices = sorted(completed_chunks)
            progress.current_document = str(chunk_entries[0].file) if chunk_entries else "n/a"
            progress.translated_entry_count = len(translated_by_key)
            progress.total_entry_count = len(entries)
            progress.elapsed_time_seconds = perf_counter() - started_at
            progress.translated_entries = [
                self._entry_to_progress(translated_by_key[self._entry_key(entry)])
                for entry in entries
                if self._entry_key(entry) in translated_by_key
            ]
            self._save_progress_atomic(progress_path, progress)

        translated_entries: list[Any] = []
        for entry in entries:
            key = self._entry_key(entry)
            translated = translated_by_key.get(key)
            if translated is not None:
                translated_entries.append(translated)

        if hasattr(self.translator, "__dict__"):
            self.translator.__dict__.pop("_resume_diagnostics_chunk_index", None)
            self.translator.__dict__.pop("_resume_diagnostics_loaded_entries", None)
            self.translator.__dict__.pop("_resume_diagnostics_active", None)

        return translated_entries

    def _build_input_signature(
        self,
        *,
        entries: list[Any],
        only_file: str | None,
        limit: int | None,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"only_file={only_file}|limit={limit}|count={len(entries)}\n".encode("utf-8"))
        for entry in entries:
            hasher.update(str(entry.file).encode("utf-8"))
            hasher.update(b"\n")
            hasher.update(repr(entry.path).encode("utf-8"))
            hasher.update(b"\n")
            hasher.update(entry.field.encode("utf-8"))
            hasher.update(b"\n")
            hasher.update(entry.source.encode("utf-8"))
            hasher.update(b"\n")
        return hasher.hexdigest()

    def _entry_key(self, entry: Any) -> tuple[str, tuple[Any, ...], str]:
        return (str(entry.file), tuple(entry.path), entry.field)

    def _entry_to_progress(self, entry: Any) -> dict[str, Any]:
        return {
            "file": str(entry.file),
            "path": entry.path,
            "field": entry.field,
            "source": entry.source,
        }

    def _entry_from_progress(self, payload: dict[str, Any]) -> Any:
        from .scanner import TranslationEntry

        return TranslationEntry(
            file=Path(payload["file"]),
            path=list(payload["path"]),
            field=str(payload["field"]),
            source=str(payload["source"]),
        )

    def _load_progress(self, progress_path: Path) -> ResumeProgress | None:
        if not progress_path.exists():
            return None
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupted progress file: {progress_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Corrupted progress file: {progress_path}")
        return ResumeProgress.from_dict(payload)

    def _save_progress_atomic(self, progress_path: Path, progress: ResumeProgress) -> None:
        temp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(progress.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(progress_path)

    def _filter_single_document(
        self,
        input_path: Path,
        documents: list[Any],
        entries: list[Any],
        only_file: str,
    ) -> tuple[list[Any], list[Any]]:
        selected_documents = [document for document in documents if self._document_matches_only_file(input_path, document.path, only_file)]
        if not selected_documents:
            raise FileNotFoundError(f"No document matched only-file: {only_file}")

        selected_document_paths = {document.path for document in selected_documents}
        selected_entries = [entry for entry in entries if entry.file in selected_document_paths]
        return selected_documents, selected_entries

    def _document_matches_only_file(self, input_path: Path, document_path: Path, only_file: str) -> bool:
        if document_path.name == only_file:
            return True

        try:
            relative_path = document_path.relative_to(input_path)
        except ValueError:
            return False

        return relative_path.as_posix() == only_file or relative_path.name == only_file
