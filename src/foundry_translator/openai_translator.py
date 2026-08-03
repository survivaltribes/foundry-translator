"""OpenAI-backed text translator using the Responses API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import time
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from openai import APIConnectionError
from openai import APITimeoutError
from openai import APIStatusError
from openai import OpenAI

from .protect import Protect
from .protect import ProtectedText
from .scanner import TranslationEntry


class OpenAITranslatorError(RuntimeError):
    """Base exception for translator failures."""


class OpenAITranslatorRequestError(OpenAITranslatorError):
    """Raised when the OpenAI request fails."""


class OpenAITranslatorTimeoutError(OpenAITranslatorRequestError):
    """Raised when the OpenAI request exceeds the configured timeout."""


class OpenAITranslatorCountError(OpenAITranslatorError):
    """Raised when the API returns an unexpected number of translations."""


@dataclass(slots=True)
class RestoreReplayArtifacts:
    """Saved diagnostics used to replay a restore failure."""

    debug_dir: Path
    original_source: str
    protected_source: str
    translated_protected: str
    placeholders_before_restore: list[str]
    placeholders_after_translation: list[str]
    file_name: str
    field_name: str
    json_path: str
    restored_attempt: str | None


TRANSLATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["translations"],
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "translation"],
                "properties": {
                    "id": {"type": "integer"},
                    "translation": {"type": "string"},
                },
            },
        },
    },
}


class OpenAITranslator:
    """Translate text batches through the official OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-5.5",
        target_language: str = "French",
        batch_size: int = 20,
        max_prompt_chars: int = 40000,
        max_retries: int = 3,
        timeout: float = 180.0,
        retry_delay: float = 0.5,
        retry_backoff_factor: float = 2.0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        client: OpenAI | None = None,
        logger: logging.Logger | None = None,
        cache: Any = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("An OpenAI API key is required")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be greater than zero")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")

        self.model = model
        self.target_language = target_language
        self.batch_size = batch_size
        self.max_prompt_chars = max_prompt_chars
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.temperature = temperature
        self.top_p = top_p
        self.client = client or OpenAI(api_key=self.api_key, timeout=self.timeout)
        self.logger = logger or logging.getLogger(__name__)
        self.cache = cache
        self.protector = Protect()
        self._progress_reporter: Any | None = None
        self._progress_context: dict[str, Any] | None = None
        self._last_prompt_for_count_error: str = ""
        self._last_response_for_count_error: str = ""

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
        glossary: dict[str, str] | None = None,
    ) -> list[str]:
        """Translate a list of texts while preserving input order."""

        if not isinstance(texts, list):
            raise TypeError("texts must be provided as a list")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("all entries in texts must be strings")
        if not source_language or not target_language:
            raise ValueError("source_language and target_language must be non-empty")
        if glossary is not None and not isinstance(glossary, dict):
            raise TypeError("glossary must be a dictionary or None")

        if not texts:
            return []

        batches = self._split_texts_into_prompt_batches(
            texts,
            source_language=source_language,
            target_language=target_language,
            glossary=glossary,
        )
        self._progress_context = {
            "current_batch": 0,
            "total_batches": len(batches),
            "translated_count": 0,
            "total_texts": len(texts),
            "current_file": None,
        }

        translated: list[str] = []
        for batch_number, batch in enumerate(batches, start=1):
            self._progress_context["current_batch"] = batch_number
            translated.extend(
                self._translate_batch(
                    batch,
                    source_language=source_language,
                    target_language=target_language,
                    glossary=glossary,
                )
            )
            self._progress_context["translated_count"] = len(translated)

        if len(translated) != len(texts):
            prompt = self._last_prompt_for_count_error
            response_text = self._last_response_for_count_error
            prompt_path, response_path = self._persist_failed_count_debug_artifacts(
                prompt=prompt,
                response_text=response_text,
            )
            self.logger.warning(
                "translation count mismatch after all batches",
                extra={
                    "expected_translation_count": len(texts),
                    "received_translation_count": len(translated),
                    "batch_number": int((self._progress_context or {}).get("current_batch", 0)),
                    "prompt_length": len(prompt),
                    "prompt_path": str(prompt_path),
                    "response_path": str(response_path),
                },
            )
            raise OpenAITranslatorCountError(
                f"Expected {len(texts)} translations but received {len(translated)}"
            )

        return translated

    def translate(self, entries: list[TranslationEntry]) -> list[TranslationEntry]:
        """Compatibility wrapper for legacy callers that use translation entries."""

        if not entries:
            return []

        results: list[TranslationEntry | None] = [None] * len(entries)
        pending_groups: list[tuple[str, list[tuple[int, TranslationEntry]], Any]] = []
        pending_sources: dict[str, int] = {}

        for index, entry in enumerate(entries):
            if self.cache is not None and hasattr(self.cache, "get"):
                cached = self.cache.get(entry.source)
                if cached is not None:
                    results[index] = replace(entry, source=cached)
                    continue

            if entry.source in pending_sources:
                pending_groups[pending_sources[entry.source]][1].append((index, entry))
                continue

            pending_sources[entry.source] = len(pending_groups)
            pending_groups.append((entry.source, [(index, entry)], self.protector.protect(entry.source)))

        if not pending_groups:
            return [result for result in results if result is not None]

        batches = self._split_pending_groups_into_prompt_batches(pending_groups)
        self._progress_context = {
            "current_batch": 0,
            "total_batches": len(batches),
            "translated_count": 0,
            "total_texts": len(pending_groups),
            "current_file": None,
        }

        translated_protected_texts: list[str] = []
        for batch_number, chunk in enumerate(batches, start=1):
            self._progress_context["current_batch"] = batch_number
            self._progress_context["current_file"] = None
            if chunk:
                first_group = chunk[0]
                matching_entries = first_group[1]
                if matching_entries:
                    self._progress_context["current_file"] = str(matching_entries[0][1].file)
            chunk_texts = [protected.protected for _, _, protected in chunk]
            translated_chunk = self._translate_batch(
                chunk_texts,
                source_language="English",
                target_language=self.target_language,
            )
            translated_protected_texts.extend(translated_chunk)
            self._progress_context["translated_count"] = len(translated_protected_texts)

        for (source, matching_entries, protected_text), translated_protected_text in zip(
            pending_groups,
            translated_protected_texts,
            strict=True,
        ):
            entry_index, entry = matching_entries[0]
            try:
                restored_source = self._restore_protected_text(protected_text, translated_protected_text)
            except ValueError as exc:
                if str(exc) != "Duplicate placeholders detected in protected text":
                    raise

                translated_attempt = self._inject_translation_into_masked_text(
                    protected_text.protected,
                    translated_protected_text,
                )
                original_placeholders = list(protected_text.placeholders.keys())
                translated_placeholders = self._extract_placeholders_from_text(translated_attempt)
                debug_dir = self._persist_restore_duplicate_debug_artifacts(
                    original_source=protected_text.original,
                    protected_source=protected_text.protected,
                    translated_protected=translated_protected_text,
                    placeholders_before_restore=original_placeholders,
                    placeholders_after_translation=translated_placeholders,
                    file_name=entry.file.name,
                    field_name=entry.field,
                    json_path=entry.path,
                    restored_attempt=translated_attempt,
                )
                self.logger.error(
                    "duplicate placeholders detected during restore",
                    extra={
                        "entry_id": entry_index + 1,
                        "field": entry.field,
                        "file": str(entry.file),
                        "json_path": self._render_json_path(entry.path),
                        "original_placeholders": original_placeholders,
                        "translated_placeholders": translated_placeholders,
                        "debug_dir": str(debug_dir),
                    },
                )
                raise

            if self.cache is not None and hasattr(self.cache, "put"):
                self.cache.put(source, restored_source)
                if hasattr(self.cache, "save"):
                    self.cache.save()
            for index, entry in matching_entries:
                results[index] = replace(entry, source=restored_source)

        return [result for result in results if result is not None]

    def _split_texts_into_prompt_batches(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
        glossary: dict[str, str] | None = None,
    ) -> list[list[str]]:
        if not texts:
            return []

        batches: list[list[str]] = []
        current_batch: list[str] = []
        for text in texts:
            candidate_batch = [*current_batch, text]
            candidate_prompt = self._build_prompt(
                candidate_batch,
                source_language=source_language,
                target_language=target_language,
                glossary=glossary,
            )
            if current_batch and len(candidate_prompt) > self.max_prompt_chars:
                batches.append(current_batch)
                current_batch = [text]
            else:
                current_batch = candidate_batch

        if current_batch:
            batches.append(current_batch)

        return batches

    def _split_pending_groups_into_prompt_batches(
        self,
        pending_groups: list[tuple[str, list[tuple[int, TranslationEntry]], Any]],
    ) -> list[list[tuple[str, list[tuple[int, TranslationEntry]], Any]]]:
        if not pending_groups:
            return []

        batches: list[list[tuple[str, list[tuple[int, TranslationEntry]], Any]]] = []
        current_batch: list[tuple[str, list[tuple[int, TranslationEntry]], Any]] = []
        for group in pending_groups:
            candidate_batch = [*current_batch, group]
            candidate_prompt = self._build_prompt(
                [entry.protected for _, _, entry in candidate_batch],
                source_language="English",
                target_language=self.target_language,
            )
            if current_batch and len(candidate_prompt) > self.max_prompt_chars:
                batches.append(current_batch)
                current_batch = [group]
            else:
                current_batch = candidate_batch

        if current_batch:
            batches.append(current_batch)

        return batches

    def _translate_batch(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
        glossary: dict[str, str] | None = None,
    ) -> list[str]:
        if not texts:
            return []

        return self._translate_with_resilience(
            texts,
            source_language=source_language,
            target_language=target_language,
            glossary=glossary,
            batch_number=1,
            total_batches=1,
        )

    def _translate_with_resilience(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
        glossary: dict[str, str] | None = None,
        batch_number: int = 1,
        total_batches: int = 1,
    ) -> list[str]:
        if not texts:
            return []

        prompt = self._build_prompt(
            texts,
            source_language=source_language,
            target_language=target_language,
            glossary=glossary,
        )
        requested_ids = [index for index, _ in enumerate(texts, start=1)]
        self._log_batch_start(texts, batch_number=batch_number, total_batches=total_batches, prompt=prompt)

        for attempt in range(1, self.max_retries + 1):
            response_text: str | None = None
            try:
                started_at = time.perf_counter()
                response_text = self._call_openai(prompt)
                self._last_prompt_for_count_error = prompt
                self._last_response_for_count_error = response_text
                duration = time.perf_counter() - started_at
                translations = self._parse_response(response_text, requested_ids=requested_ids)
                self.logger.info(
                    "translations completed",
                    extra={
                        "batch_number": batch_number,
                        "total_batches": total_batches,
                        "batch_size": len(texts),
                        "attempt": attempt,
                        "duration_seconds": round(duration, 6),
                        "model": self.model,
                    },
                )
                return translations
            except OpenAITranslatorCountError as exc:
                expected_count, received_count = self._extract_translation_counts(exc)
                prompt_path, response_path = self._persist_failed_count_debug_artifacts(
                    prompt=prompt,
                    response_text=response_text or "",
                )
                self.logger.warning(
                    "translation response parse/validation failed",
                    extra={
                        "batch_number": batch_number,
                        "total_batches": total_batches,
                        "batch_size": len(texts),
                        "attempt": attempt,
                        "error": str(exc),
                        "prompt_path": str(prompt_path),
                        "response_path": str(response_path),
                    },
                )
                if attempt >= self.max_retries:
                    raise
                delay = self.retry_delay * (self.retry_backoff_factor ** (attempt - 1))
                self.logger.warning(
                    "translation response invalid; retrying",
                    extra={
                        "expected_translation_count": expected_count,
                        "received_translation_count": received_count,
                        "batch_number": batch_number,
                        "prompt_length": len(prompt),
                        "prompt_path": str(prompt_path),
                        "response_path": str(response_path),
                        "total_batches": total_batches,
                        "batch_size": len(texts),
                        "attempt": attempt,
                        "delay_seconds": delay,
                        "error": str(exc),
                    },
                )
                time.sleep(delay)
            except (APITimeoutError, APIConnectionError, APIStatusError) as exc:  # type: ignore[misc]
                self._log_openai_exception(exc, batch_number=batch_number, total_batches=total_batches)
                if len(texts) <= 1:
                    raise OpenAITranslatorTimeoutError(
                        f"OpenAI request failed after {self.max_retries} attempts"
                    ) from exc
                midpoint = len(texts) // 2
                if midpoint < 1:
                    raise OpenAITranslatorTimeoutError(
                        f"OpenAI request failed after {self.max_retries} attempts"
                    ) from exc
                left_batch = texts[:midpoint]
                right_batch = texts[midpoint:]
                self.logger.warning(
                    "splitting batch after transport failure",
                    extra={
                        "batch_number": batch_number,
                        "total_batches": total_batches,
                        "batch_size": len(texts),
                        "split_into": [len(left_batch), len(right_batch)],
                        "error_type": exc.__class__.__name__,
                    },
                )
                left_results = self._translate_with_resilience(
                    left_batch,
                    source_language=source_language,
                    target_language=target_language,
                    glossary=glossary,
                    batch_number=batch_number,
                    total_batches=total_batches + 1,
                )
                right_results = self._translate_with_resilience(
                    right_batch,
                    source_language=source_language,
                    target_language=target_language,
                    glossary=glossary,
                    total_batches=total_batches + 1,
                    batch_number=batch_number + 1,
                )
                return [*left_results, *right_results]
            except Exception as exc:  # pragma: no cover - defensive guard
                self._log_openai_exception(exc, batch_number=batch_number, total_batches=total_batches)
                if isinstance(exc, OpenAITranslatorRequestError):
                    raise
                raise OpenAITranslatorRequestError("OpenAI request failed") from exc

        raise OpenAITranslatorRequestError("OpenAI request failed")

    def _call_openai(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "translation_batch",
                    "strict": True,
                    "schema": TRANSLATION_RESPONSE_SCHEMA,
                }
            },
        }
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout

        batch_size = len(self._extract_input_items_from_prompt(prompt))
        prompt_size = len(prompt.encode("utf-8"))
        started_at = time.perf_counter()
        try:
            response = self.client.responses.create(**kwargs)
        except Exception as exc:  # pragma: no cover - defensive guard
            elapsed = time.perf_counter() - started_at
            self._report_progress(
                prompt=prompt,
                elapsed_seconds=elapsed,
                batch_size=batch_size,
                prompt_size=prompt_size,
                failed=True,
                exception=exc,
            )
            raise

        elapsed = time.perf_counter() - started_at
        output_text = getattr(response, "output_text", "")
        self._report_progress(
            prompt=prompt,
            elapsed_seconds=elapsed,
            batch_size=batch_size,
            prompt_size=prompt_size,
            failed=False,
            output_text=output_text,
        )
        return output_text

    def _report_progress(
        self,
        *,
        prompt: str,
        elapsed_seconds: float,
        batch_size: int,
        prompt_size: int,
        failed: bool,
        exception: Exception | None = None,
        output_text: str | None = None,
    ) -> None:
        reporter = getattr(self, "_progress_reporter", None)
        if reporter is None:
            return

        context = self._progress_context or {}
        batch_number = int(context.get("current_batch", 0))
        total_batches = int(context.get("total_batches", 0))
        translated_count = int(context.get("translated_count", 0))
        total_texts = int(context.get("total_texts", 0))
        current_file = context.get("current_file")
        if not failed:
            translated_count += batch_size
            self._progress_context = {
                **context,
                "translated_count": translated_count,
            }
        reporter.on_request_completed(
            batch_number=batch_number,
            total_batches=total_batches,
            batch_size=batch_size,
            prompt_size=prompt_size,
            elapsed_seconds=elapsed_seconds,
            current_file=current_file,
            translated_count=translated_count,
            total_texts=total_texts,
            failed=failed,
        )

    def _log_batch_start(self, texts: list[str], *, batch_number: int, total_batches: int, prompt: str) -> None:
        self.logger.info(
            "requesting translations",
            extra={
                "batch_number": batch_number,
                "total_batches": total_batches,
                "batch_size": len(texts),
                "estimated_prompt_size": len(prompt.encode("utf-8")),
                "model": self.model,
            },
        )

    def _log_batch_success(self, texts: list[str], *, batch_number: int, total_batches: int) -> None:
        self.logger.info(
            "translations completed",
            extra={
                "batch_number": batch_number,
                "total_batches": total_batches,
                "batch_size": len(texts),
                "model": self.model,
            },
        )

    def _log_openai_exception(self, exc: Exception, *, batch_number: int, total_batches: int) -> None:
        status_code = None
        message = str(exc)
        if hasattr(exc, "status_code"):
            status_code = getattr(exc, "status_code")
        elif hasattr(exc, "response") and getattr(exc, "response", None) is not None:
            status_code = getattr(exc.response, "status_code", None)

        self.logger.error(
            "OpenAI request failed",
            extra={
                "batch_number": batch_number,
                "total_batches": total_batches,
                "exception_class": exc.__class__.__name__,
                "status_code": status_code,
                "error_message": message,
            },
        )

    def _extract_translation_counts(self, exc: OpenAITranslatorCountError) -> tuple[int | None, int | None]:
        match = re.search(r"Expected (\d+) translations but received (\d+)", str(exc))
        if not match:
            return None, None
        return int(match.group(1)), int(match.group(2))


    @staticmethod
    def strip_appended_original_protected_source(translated_text: str, protected_source: str) -> str:
        """Remove duplicated protected-source suffixes before restore.

        This strips either:
        - an exact trailing copy of the full protected source, or
        - a trailing copy that starts at any placeholder boundary, or
        - text from the first repeated source placeholder onward.
        """

        if not translated_text or not protected_source:
            return translated_text

        if translated_text.endswith(protected_source):
            return translated_text[: -len(protected_source)]

        placeholder_pattern = re.compile(r"__FT_[A-Z_]+_\d{5}__")
        protected_placeholders = {
            match.group(0)
            for match in placeholder_pattern.finditer(protected_source)
        }

        for match in placeholder_pattern.finditer(protected_source):
            candidate_suffix = protected_source[match.start() :]
            if translated_text.endswith(candidate_suffix):
                suffix_start = len(translated_text) - len(candidate_suffix)
                first_suffix_placeholder = match.group(0)
                has_prior_same_placeholder = any(
                    m.group(0) == first_suffix_placeholder and m.start() < suffix_start
                    for m in placeholder_pattern.finditer(translated_text)
                )
                if has_prior_same_placeholder:
                    return translated_text[: -len(candidate_suffix)]

        seen_placeholders: set[str] = set()
        for match in placeholder_pattern.finditer(translated_text):
            placeholder = match.group(0)
            if placeholder not in protected_placeholders:
                continue
            if placeholder in seen_placeholders:
                return translated_text[: match.start()]
            seen_placeholders.add(placeholder)

        return translated_text


    @staticmethod
    def load_restore_replay_artifacts(debug_dir: Path | str) -> RestoreReplayArtifacts:
        """Load restore-failure diagnostics from a debug directory."""

        debug_path = Path(debug_dir).expanduser().resolve()

        def read_text(name: str) -> str:
            return (debug_path / name).read_text(encoding="utf-8")

        def read_json_list(name: str) -> list[str]:
            payload = json.loads(read_text(name))
            if not isinstance(payload, list):
                raise ValueError(f"Expected {name} to contain a JSON array")
            return [str(item) for item in payload]

        restored_attempt_path = debug_path / "restored_attempt.txt"
        restored_attempt = restored_attempt_path.read_text(encoding="utf-8") if restored_attempt_path.exists() else None

        return RestoreReplayArtifacts(
            debug_dir=debug_path,
            original_source=read_text("original_source.txt"),
            protected_source=read_text("protected_source.txt"),
            translated_protected=read_text("translated_protected.txt"),
            placeholders_before_restore=read_json_list("placeholders_before_restore.json"),
            placeholders_after_translation=read_json_list("placeholders_after_translation.json"),
            file_name=read_text("file_name.txt"),
            field_name=read_text("field_name.txt"),
            json_path=read_text("json_path.txt"),
            restored_attempt=restored_attempt,
        )


    @staticmethod
    def replay_restore_from_debug_dir(debug_dir: Path | str) -> str:
        """Replay a restore failure from saved diagnostics without making an OpenAI call."""

        artifacts = OpenAITranslator.load_restore_replay_artifacts(debug_dir)
        translator = OpenAITranslator(
            api_key="debug-key",
            model="gpt-4.1-mini",
            target_language="French",
            batch_size=1,
            client=object(),
        )
        protected_text = ProtectedText(
            original=artifacts.original_source,
            protected=artifacts.protected_source,
            placeholders=translator.protector.protect(artifacts.original_source).placeholders,
        )
        return translator._restore_protected_text(protected_text, artifacts.translated_protected)

    def _get_debug_artifact_paths(self) -> tuple[Path, Path]:
        project_root = Path(__file__).resolve().parents[2]
        debug_dir = project_root / "debug"
        return debug_dir / "failed_prompt.txt", debug_dir / "failed_response.txt"

    def _get_restore_debug_artifact_dir(self) -> Path:
        project_root = Path(__file__).resolve().parents[2]
        debug_dir = project_root / "debug"
        return debug_dir / f"restore_duplicate_placeholders_{time.time_ns()}"

    def _persist_failed_count_debug_artifacts(self, *, prompt: str, response_text: str) -> tuple[Path, Path]:
        prompt_path, response_path = self._get_debug_artifact_paths()
        try:
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")
            response_path.write_text(response_text, encoding="utf-8")
            self.logger.info(
                "saved OpenAI response debug artifacts",
                extra={
                    "prompt_path": str(prompt_path.resolve()),
                    "response_path": str(response_path.resolve()),
                    "prompt_length": len(prompt),
                    "response_length": len(response_text),
                },
            )
        except OSError as exc:
            self.logger.warning(
                "failed to persist OpenAI count mismatch debug artifacts",
                extra={
                    "prompt_path": str(prompt_path.resolve()),
                    "response_path": str(response_path.resolve()),
                    "error": str(exc),
                },
            )
        return prompt_path.resolve(), response_path.resolve()

    def _persist_restore_duplicate_debug_artifacts(
        self,
        *,
        original_source: str,
        protected_source: str,
        translated_protected: str,
        placeholders_before_restore: list[str],
        placeholders_after_translation: list[str],
        file_name: str,
        field_name: str,
        json_path: list[str | int],
        restored_attempt: str | None,
    ) -> Path:
        debug_dir = self._get_restore_debug_artifact_dir()
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "original_source.txt").write_text(original_source, encoding="utf-8")
            (debug_dir / "protected_source.txt").write_text(protected_source, encoding="utf-8")
            (debug_dir / "translated_protected.txt").write_text(translated_protected, encoding="utf-8")
            (debug_dir / "placeholders_before_restore.json").write_text(
                json.dumps(placeholders_before_restore, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (debug_dir / "placeholders_after_translation.json").write_text(
                json.dumps(placeholders_after_translation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (debug_dir / "file_name.txt").write_text(file_name, encoding="utf-8")
            (debug_dir / "field_name.txt").write_text(field_name, encoding="utf-8")
            (debug_dir / "json_path.txt").write_text(self._render_json_path(json_path), encoding="utf-8")
            if restored_attempt is not None:
                (debug_dir / "restored_attempt.txt").write_text(restored_attempt, encoding="utf-8")
            self.logger.info(
                "saved restore duplicate placeholder debug artifacts",
                extra={
                    "debug_dir": str(debug_dir.resolve()),
                },
            )
        except OSError as exc:
            self.logger.warning(
                "failed to persist restore duplicate placeholder debug artifacts",
                extra={
                    "error": str(exc),
                    "debug_dir": str(debug_dir.resolve()),
                },
            )
        return debug_dir.resolve()

    def _render_json_path(self, path_parts: list[str | int]) -> str:
        result = "$"
        for part in path_parts:
            if isinstance(part, int):
                result += f"[{part}]"
            else:
                escaped = str(part).replace("'", "\\'")
                result += f"['{escaped}']"
        return result

    def _extract_placeholders_from_text(self, text: str) -> list[str]:
        placeholder_pattern = getattr(self.protector, "_PLACEHOLDER_PATTERN", re.compile(r"__FT_[A-Z_]+_\d{5}__"))
        return [match.group(0) for match in placeholder_pattern.finditer(text)]

    def _restore_protected_text(self, protected_text: Any, translated_text: str) -> str:
        """Restore protected markers into translated text while preserving surrounding content."""

        sanitized_translated_text = self._strip_appended_original_protected_source(
            translated_text=translated_text,
            protected_source=protected_text.protected,
        )

        if not sanitized_translated_text:
            return sanitized_translated_text

        placeholder_pattern = getattr(self.protector, "_PLACEHOLDER_PATTERN", re.compile(r"__FT_[A-Z_]+_\d{5}__"))
        matches = list(placeholder_pattern.finditer(protected_text.protected))
        if not matches:
            return sanitized_translated_text

        translated_placeholders = self._extract_placeholders_from_text(sanitized_translated_text)
        if translated_placeholders:
            return self.protector.restore(protected_text, sanitized_translated_text)

        translated_masked_text = self._inject_translation_into_masked_text(
            protected_text.protected,
            sanitized_translated_text,
        )
        return self.protector.restore(protected_text, translated_masked_text)

    def _strip_appended_original_protected_source(self, *, translated_text: str, protected_source: str) -> str:
        return self.__class__.strip_appended_original_protected_source(translated_text, protected_source)

    def _inject_translation_into_masked_text(self, masked_text: str, translated_text: str) -> str:
        placeholder_pattern = getattr(self.protector, "_PLACEHOLDER_PATTERN", re.compile(r"__FT_[A-Z_]+_\d{5}__"))
        matches = list(placeholder_pattern.finditer(masked_text))
        if not matches:
            return translated_text

        result: list[str] = []
        last = 0
        replaced = False
        for match in matches:
            segment = masked_text[last : match.start()]
            if not replaced and segment.strip():
                leading = segment[: len(segment) - len(segment.lstrip())]
                trailing = segment[len(segment.rstrip()) :]
                segment = leading + translated_text + trailing
                replaced = True
            result.append(segment)
            result.append(match.group(0))
            last = match.end()

        tail = masked_text[last:]
        if not replaced and tail.strip():
            leading = tail[: len(tail) - len(tail.lstrip())]
            trailing = tail[len(tail.rstrip()) :]
            tail = leading + translated_text + trailing

        result.append(tail)
        return "".join(result)

    def _build_request_items(self, texts: list[str]) -> list[dict[str, Any]]:
        return [{"id": index, "text": text} for index, text in enumerate(texts, start=1)]

    def _extract_input_items_from_prompt(self, prompt: str) -> list[dict[str, Any]]:
        marker = "Inputs JSON:"
        if marker not in prompt:
            return []

        payload = prompt.split(marker, 1)[1].strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []

        request_items: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            identifier = item.get("id")
            text = item.get("text")
            if isinstance(identifier, bool) or not isinstance(identifier, int):
                continue
            if not isinstance(text, str):
                continue
            request_items.append({"id": identifier, "text": text})
        return request_items

    def _build_prompt(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
        glossary: dict[str, str] | None = None,
    ) -> str:
        instructions = [
            f"Translate the following texts from {source_language} to {target_language}.",
            "Return only a JSON object with a translations array.",
            "Use exactly the same ids provided in the input JSON.",
            "Keep placeholders, markup, and code-like tokens unchanged.",
            "Use deterministic, concise, natural phrasing.",
        ]

        if glossary:
            instructions.append("Glossary:")
            for term, translation in sorted(glossary.items()):
                instructions.append(f"{term} -> {translation}")

        lines = ["\n".join(instructions), "", "Inputs JSON:"]
        lines.append(json.dumps(self._build_request_items(texts), ensure_ascii=False, indent=2))

        return "\n".join(lines)

    def _parse_response(self, response_text: str, *, requested_ids: list[int]) -> list[str]:
        if not response_text:
            raise OpenAITranslatorCountError(f"Expected {len(requested_ids)} translations but received 0")

        text = response_text.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OpenAITranslatorCountError(
                f"Response was not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
            ) from exc
        if not isinstance(parsed, dict):
            raise OpenAITranslatorCountError("Expected a JSON object with a translations array")
        translations = parsed.get("translations")
        if not isinstance(translations, list):
            raise OpenAITranslatorCountError("Expected response.translations to be a JSON array")

        translations_by_id: dict[int, str] = {}
        duplicate_ids: list[int] = []
        unexpected_ids: list[int] = []
        requested_set = set(requested_ids)

        for index, item in enumerate(translations, start=1):
            if not isinstance(item, dict):
                raise OpenAITranslatorCountError(
                    f"Response item at index {index} must be an object with id and translation"
                )

            identifier = item.get("id")
            translation = item.get("translation")
            if isinstance(identifier, bool) or not isinstance(identifier, int):
                raise OpenAITranslatorCountError(f"Response item at index {index} has non-integer id")
            if not isinstance(translation, str):
                raise OpenAITranslatorCountError(
                    f"Response item for id {identifier} has non-string translation"
                )

            if identifier in translations_by_id:
                duplicate_ids.append(identifier)
            translations_by_id[identifier] = translation

            if identifier not in requested_set:
                unexpected_ids.append(identifier)

        missing_ids = [identifier for identifier in requested_ids if identifier not in translations_by_id]
        if duplicate_ids:
            duplicates = sorted(set(duplicate_ids))
            raise OpenAITranslatorCountError(f"Duplicate translation ids in response: {duplicates}")
        if unexpected_ids:
            extras = sorted(set(unexpected_ids))
            raise OpenAITranslatorCountError(f"Unexpected translation ids in response: {extras}")
        if missing_ids:
            raise OpenAITranslatorCountError(f"Missing translation ids in response: {missing_ids}")

        return [translations_by_id[identifier] for identifier in requested_ids]
