"""OpenAI-backed text translator using the Responses API."""

from __future__ import annotations

import json
import hashlib
import logging
import os
from collections import Counter
from pathlib import Path
import re
import time
from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
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


class PlaceholderMismatchError(OpenAITranslatorError):
    """Raised when translated placeholders do not match protected placeholders."""


@dataclass(slots=True)
class RestoreReplayArtifacts:
    """Saved diagnostics used to replay a restore failure."""

    debug_dir: Path
    original_source: str
    protected_source: str
    translated_protected: str
    sanitized_translated: str
    exception: str | None
    placeholders_before_restore: list[str]
    placeholders_after_restore: list[str]
    file_name: str
    field_name: str
    json_path: str
    restored_attempt: str | None


@dataclass(slots=True)
class PlaceholderMismatchDetails:
    """Details about a placeholder mismatch detected before restore."""

    chunk_index: int
    entry_index: int
    entry: TranslationEntry
    protected_text: ProtectedText
    translated_protected_text: str
    expected_placeholders: list[str]
    actual_placeholders: list[str]
    missing_placeholders: list[str]
    unexpected_placeholders: list[str]


class MismatchDecisionAction(str, Enum):
    """Decision emitted by mismatch policy."""

    FATAL = "fatal"
    WARNING = "warning"


@dataclass(slots=True)
class MismatchDecision:
    """Outcome of placeholder mismatch policy evaluation."""

    action: MismatchDecisionAction
    missing_placeholders: list[str]
    rule: str | None = None
    severity: str | None = None
    reason: str | None = None


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

    _TRACE_PLACEHOLDER_NAME = "__FT_UUID_00006__"

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
        self._last_response_metadata_for_count_error: dict[str, Any] = {}

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
                response_metadata=self._last_response_metadata_for_count_error,
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
        translated_protected_origins: list[str] = []
        tolerated_protected_overrides: dict[str, ProtectedText] = {}
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
            translated_chunk_origins = ["fresh"] * len(translated_chunk)

            if getattr(self, "_resume_diagnostics_active", False):
                self.logger.info(
                    "resume diagnostics: translated_protected source=fresh batch=%s chunk_index=%s count=%s",
                    batch_number,
                    getattr(self, "_resume_diagnostics_chunk_index", "n/a"),
                    len(translated_chunk),
                )

            mismatches = self._collect_placeholder_mismatches(chunk, translated_chunk)
            if mismatches:
                self.logger.warning(
                    "placeholder mismatch detected after translation; retrying once",
                    extra={
                        "batch_number": batch_number,
                        "total_batches": len(batches),
                        "mismatch_count": len(mismatches),
                        "trace_placeholder": self._TRACE_PLACEHOLDER_NAME,
                    },
                )
                retry_positions = [mismatch.chunk_index for mismatch in mismatches]
                retry_texts = [chunk[position][2].protected for position in retry_positions]
                retried_texts = self._translate_batch(
                    retry_texts,
                    source_language="English",
                    target_language=self.target_language,
                )
                for position, translated_text in zip(retry_positions, retried_texts, strict=True):
                    translated_chunk[position] = translated_text
                    translated_chunk_origins[position] = "fresh_retry"
                mismatches = self._collect_placeholder_mismatches(chunk, translated_chunk)
                if mismatches:
                    fatal_mismatches: list[PlaceholderMismatchDetails] = []
                    for mismatch in mismatches:
                        decision = self._decide_mismatch_after_retry(mismatch)
                        if decision.action == MismatchDecisionAction.FATAL:
                            fatal_mismatches.append(mismatch)
                            continue

                        source = chunk[mismatch.chunk_index][0]
                        tolerated_protected_overrides[source] = self._without_tolerated_placeholders(
                            mismatch.protected_text,
                            decision.missing_placeholders,
                        )
                        self.logger.warning(
                            "tolerated missing placeholders; continuing translation",
                            extra={
                                "rule": decision.rule,
                                "severity": decision.severity,
                                "placeholders": decision.missing_placeholders,
                                "reason": decision.reason,
                                "file": str(mismatch.entry.file),
                                "field": mismatch.entry.field,
                                "json_path": self._render_json_path(mismatch.entry.path),
                                "pipeline_continues": True,
                            },
                        )

                    if fatal_mismatches:
                        raise self._raise_placeholder_mismatch_after_retry(fatal_mismatches)

            translated_protected_texts.extend(translated_chunk)
            translated_protected_origins.extend(translated_chunk_origins)
            self._progress_context["translated_count"] = len(translated_protected_texts)

        first_restore_logged = False
        for (source, matching_entries, protected_text), translated_protected_text, translated_origin in zip(
            pending_groups,
            translated_protected_texts,
            translated_protected_origins,
            strict=True,
        ):
            protected_text_for_restore = tolerated_protected_overrides.get(source, protected_text)
            entry_index, entry = matching_entries[0]
            if getattr(self, "_resume_diagnostics_active", False) and not first_restore_logged:
                first_restore_logged = True
                self.logger.info(
                    "resume diagnostics: first call to _restore_protected_text source=%s chunk_index=%s loaded_entries_before_chunk=%s entry_id=%s file=%s path=%s",
                    translated_origin,
                    getattr(self, "_resume_diagnostics_chunk_index", "n/a"),
                    getattr(self, "_resume_diagnostics_loaded_entries", "n/a"),
                    entry_index + 1,
                    str(entry.file),
                    self._render_json_path(entry.path),
                )
            try:
                restored_source = self._restore_protected_text(protected_text_for_restore, translated_protected_text)
            except ValueError as exc:
                sanitized_translated_text, restored_attempt, _ = self._prepare_restore_attempt(
                    protected_text_for_restore,
                    translated_protected_text,
                )
                original_placeholders = list(protected_text_for_restore.placeholders.keys())
                placeholders_after_restore = self._extract_placeholders_from_text(restored_attempt)
                debug_dir = self._persist_restore_debug_artifacts(
                    exception_message=str(exc),
                    original_source=protected_text_for_restore.original,
                    protected_source=protected_text_for_restore.protected,
                    translated_protected=translated_protected_text,
                    sanitized_translated=sanitized_translated_text,
                    placeholders_before_restore=original_placeholders,
                    placeholders_after_restore=placeholders_after_restore,
                    file_name=entry.file.name,
                    field_name=entry.field,
                    json_path=entry.path,
                    restored_attempt=restored_attempt,
                )
                log_message = "restore failed during protected placeholder restoration"
                if str(exc) == "Duplicate placeholders detected in protected text":
                    log_message = "duplicate placeholders detected during restore"
                self.logger.error(
                    log_message,
                    extra={
                        "entry_id": entry_index + 1,
                        "field": entry.field,
                        "file": str(entry.file),
                        "json_path": self._render_json_path(entry.path),
                        "error": str(exc),
                        "original_placeholders": original_placeholders,
                        "placeholders_after_restore": placeholders_after_restore,
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

    def _decide_mismatch_after_retry(self, mismatch: PlaceholderMismatchDetails) -> MismatchDecision:
        if mismatch.unexpected_placeholders:
            return MismatchDecision(
                action=MismatchDecisionAction.FATAL,
                missing_placeholders=mismatch.missing_placeholders,
            )

        if self._is_e1_tolerable_missing_empty_strong_pair(mismatch):
            return MismatchDecision(
                action=MismatchDecisionAction.WARNING,
                missing_placeholders=mismatch.missing_placeholders,
                rule="E1_TOLERATED_MISSING",
                severity="WARNING",
                reason="missing tolerated placeholder pair recognized by E1",
            )

        return MismatchDecision(
            action=MismatchDecisionAction.FATAL,
            missing_placeholders=mismatch.missing_placeholders,
        )

    def _is_e1_tolerable_missing_empty_strong_pair(self, mismatch: PlaceholderMismatchDetails) -> bool:
        if len(mismatch.missing_placeholders) != 2:
            return False

        protected_mapping = mismatch.protected_text.placeholders
        originals: dict[str, str] = {}
        for placeholder_name in mismatch.missing_placeholders:
            placeholder = protected_mapping.get(placeholder_name)
            if not hasattr(placeholder, "original") or not hasattr(placeholder, "category"):
                return False
            if placeholder.category != "HTML":
                return False
            originals[placeholder_name] = str(placeholder.original)

        if set(originals.values()) != {"<strong>", "</strong>"}:
            return False

        expected_order = mismatch.expected_placeholders
        try:
            open_name = next(name for name, value in originals.items() if value == "<strong>")
            close_name = next(name for name, value in originals.items() if value == "</strong>")
            open_index = expected_order.index(open_name)
            close_index = expected_order.index(close_name)
        except (StopIteration, ValueError):
            return False

        if close_index != open_index + 1:
            return False

        masked_text = mismatch.protected_text.protected
        open_pos = masked_text.find(open_name)
        close_pos = masked_text.find(close_name)
        if open_pos == -1 or close_pos == -1:
            return False

        between = masked_text[open_pos + len(open_name):close_pos]
        return between.strip() == ""

    def _without_tolerated_placeholders(
        self,
        protected_text: ProtectedText,
        tolerated_placeholders: list[str],
    ) -> ProtectedText:
        placeholders = {
            key: value
            for key, value in protected_text.placeholders.items()
            if key not in set(tolerated_placeholders)
        }

        masked_text = protected_text.protected
        for placeholder_name in tolerated_placeholders:
            masked_text = masked_text.replace(placeholder_name, "")

        return ProtectedText(
            original=protected_text.original,
            protected=masked_text,
            placeholders=placeholders,
        )

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
            exceeds_prompt_chars = len(candidate_prompt) > self.max_prompt_chars
            exceeds_max_items = len(candidate_batch) > self.batch_size
            if current_batch and (exceeds_prompt_chars or exceeds_max_items):
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
            exceeds_prompt_chars = len(candidate_prompt) > self.max_prompt_chars
            exceeds_max_items = len(candidate_batch) > self.batch_size
            if current_batch and (exceeds_prompt_chars or exceeds_max_items):
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

        request_items = self._build_request_items(texts)
        return self._translate_request_items_with_resilience(
            request_items,
            source_language=source_language,
            target_language=target_language,
            glossary=glossary,
            batch_number=batch_number,
            total_batches=total_batches,
        )

    def _translate_request_items_with_resilience(
        self,
        request_items: list[dict[str, Any]],
        *,
        source_language: str,
        target_language: str,
        glossary: dict[str, str] | None = None,
        batch_number: int = 1,
        total_batches: int = 1,
    ) -> list[str]:
        if not request_items:
            return []

        texts = [str(item["text"]) for item in request_items]
        requested_ids = [int(item["id"]) for item in request_items]

        prompt = self._build_prompt_from_request_items(
            request_items,
            source_language=source_language,
            target_language=target_language,
            glossary=glossary,
        )
        self._log_batch_start(texts, batch_number=batch_number, total_batches=total_batches, prompt=prompt)

        for attempt in range(1, self.max_retries + 1):
            response_text: str | None = None
            try:
                started_at = time.perf_counter()
                response_text = self._call_openai(prompt)
                self._last_prompt_for_count_error = prompt
                self._last_response_for_count_error = response_text
                duration = time.perf_counter() - started_at

                translations_by_id, missing_ids = self._parse_response_translations_by_id(
                    response_text,
                    requested_ids=requested_ids,
                )

                if missing_ids and len(missing_ids) < len(requested_ids):
                    self.logger.warning(
                        "partial translation response detected; retrying missing ids only",
                        extra={
                            "batch_number": batch_number,
                            "total_batches": total_batches,
                            "batch_size": len(texts),
                            "attempt": attempt,
                            "missing_ids": missing_ids,
                            "received_ids": sorted(translations_by_id.keys()),
                        },
                    )
                    missing_request_items = [
                        item for item in request_items if int(item["id"]) in set(missing_ids)
                    ]
                    recovered_translations = self._translate_request_items_with_resilience(
                        missing_request_items,
                        source_language=source_language,
                        target_language=target_language,
                        glossary=glossary,
                        batch_number=batch_number,
                        total_batches=total_batches,
                    )
                    for missing_id, recovered_translation in zip(
                        missing_ids,
                        recovered_translations,
                        strict=True,
                    ):
                        translations_by_id[missing_id] = recovered_translation

                    unresolved_ids = [identifier for identifier in requested_ids if identifier not in translations_by_id]
                    if unresolved_ids:
                        raise OpenAITranslatorCountError(
                            f"Missing translation ids in response: {unresolved_ids}"
                        )

                elif missing_ids:
                    raise OpenAITranslatorCountError(f"Missing translation ids in response: {missing_ids}")

                translations = [translations_by_id[identifier] for identifier in requested_ids]
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
                    response_metadata=self._last_response_metadata_for_count_error,
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

        response_metadata = self._extract_response_metadata(response)
        self._last_response_metadata_for_count_error = response_metadata
        self.logger.info(
            "received OpenAI response metadata",
            extra={
                "response_status": response_metadata.get("status"),
                "response_incomplete_details": response_metadata.get("incomplete_details"),
                "response_usage": response_metadata.get("usage"),
                "response_max_output_tokens": response_metadata.get("max_output_tokens"),
                "response_output_items": len(response_metadata.get("output", []))
                if isinstance(response_metadata.get("output"), list)
                else None,
            },
        )

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

    def _extract_response_metadata(self, response: Any) -> dict[str, Any]:
        return {
            "status": self._to_jsonable(getattr(response, "status", None)),
            "incomplete_details": self._to_jsonable(getattr(response, "incomplete_details", None)),
            "usage": self._to_jsonable(getattr(response, "usage", None)),
            "output": self._to_jsonable(getattr(response, "output", None)),
            "max_output_tokens": self._to_jsonable(getattr(response, "max_output_tokens", None)),
        }

    def _to_jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return {str(key): self._to_jsonable(item) for key, item in value.items()}

        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]

        if hasattr(value, "model_dump"):
            try:
                return self._to_jsonable(value.model_dump())
            except Exception:
                pass

        if hasattr(value, "to_dict"):
            try:
                return self._to_jsonable(value.to_dict())
            except Exception:
                pass

        if hasattr(value, "dict"):
            try:
                return self._to_jsonable(value.dict())
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            try:
                return {
                    str(key): self._to_jsonable(item)
                    for key, item in value.__dict__.items()
                    if not str(key).startswith("_")
                }
            except Exception:
                pass

        return str(value)

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

        def read_optional_text(name: str) -> str | None:
            path = debug_path / name
            if not path.exists():
                return None
            return path.read_text(encoding="utf-8")

        def read_json_list_with_fallback(primary_name: str, fallback_name: str | None = None) -> list[str]:
            primary_path = debug_path / primary_name
            if primary_path.exists():
                return read_json_list(primary_name)
            if fallback_name is not None and (debug_path / fallback_name).exists():
                return read_json_list(fallback_name)
            raise FileNotFoundError(f"Missing required debug artifact: {primary_name}")

        translated_protected = read_text("translated_protected.txt")
        sanitized_translated = read_optional_text("sanitized_translated.txt")
        restored_attempt = read_optional_text("restored_attempt.txt")
        exception_text = read_optional_text("exception.txt")

        return RestoreReplayArtifacts(
            debug_dir=debug_path,
            original_source=read_text("original_source.txt"),
            protected_source=read_text("protected_source.txt"),
            translated_protected=translated_protected,
            sanitized_translated=sanitized_translated if sanitized_translated is not None else translated_protected,
            exception=exception_text,
            placeholders_before_restore=read_json_list("placeholders_before_restore.json"),
            placeholders_after_restore=read_json_list_with_fallback(
                "placeholders_after_restore.json",
                "placeholders_after_translation.json",
            ),
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
        if artifacts.exception is not None and artifacts.restored_attempt is not None:
            return translator.protector.restore(protected_text, artifacts.restored_attempt)
        return translator._restore_protected_text(protected_text, artifacts.translated_protected)

    def _get_debug_artifact_paths(self) -> tuple[Path, Path]:
        project_root = Path(__file__).resolve().parents[2]
        debug_dir = project_root / "debug"
        return debug_dir / "failed_prompt.txt", debug_dir / "failed_response.txt"

    def _get_restore_debug_artifact_dir(self) -> Path:
        project_root = Path(__file__).resolve().parents[2]
        debug_dir = project_root / "debug"
        return debug_dir / f"restore_duplicate_placeholders_{time.time_ns()}"

    def _persist_failed_count_debug_artifacts(
        self,
        *,
        prompt: str,
        response_text: str,
        response_metadata: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        prompt_path, response_path = self._get_debug_artifact_paths()
        metadata_path = response_path.with_name("failed_response_metadata.json")
        try:
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")
            response_path.write_text(response_text, encoding="utf-8")
            metadata_payload = response_metadata or self._last_response_metadata_for_count_error
            metadata_path.write_text(
                json.dumps(metadata_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.logger.info(
                "saved OpenAI response debug artifacts",
                extra={
                    "prompt_path": str(prompt_path.resolve()),
                    "response_path": str(response_path.resolve()),
                    "response_metadata_path": str(metadata_path.resolve()),
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
                    "response_metadata_path": str(metadata_path.resolve()),
                    "error": str(exc),
                },
            )
        return prompt_path.resolve(), response_path.resolve()

    def _persist_restore_debug_artifacts(
        self,
        *,
        exception_message: str,
        original_source: str,
        protected_source: str,
        translated_protected: str,
        sanitized_translated: str,
        placeholders_before_restore: list[str],
        placeholders_after_restore: list[str],
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
            (debug_dir / "sanitized_translated.txt").write_text(sanitized_translated, encoding="utf-8")
            (debug_dir / "exception.txt").write_text(exception_message, encoding="utf-8")
            (debug_dir / "placeholders_before_restore.json").write_text(
                json.dumps(placeholders_before_restore, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (debug_dir / "placeholders_after_restore.json").write_text(
                json.dumps(placeholders_after_restore, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (debug_dir / "file_name.txt").write_text(file_name, encoding="utf-8")
            (debug_dir / "field_name.txt").write_text(field_name, encoding="utf-8")
            (debug_dir / "json_path.txt").write_text(self._render_json_path(json_path), encoding="utf-8")
            if restored_attempt is not None:
                (debug_dir / "restored_attempt.txt").write_text(restored_attempt, encoding="utf-8")
            self.logger.info(
                "saved restore debug artifacts",
                extra={
                    "debug_dir": str(debug_dir.resolve()),
                },
            )
        except OSError as exc:
            self.logger.warning(
                "failed to persist restore debug artifacts",
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

    def _collect_placeholder_mismatches(
        self,
        chunk: list[tuple[str, list[tuple[int, TranslationEntry]], Any]],
        translated_chunk: list[str],
    ) -> list[PlaceholderMismatchDetails]:
        mismatches: list[PlaceholderMismatchDetails] = []
        for chunk_index, ((_source, matching_entries, protected_text), translated_protected_text) in enumerate(
            zip(chunk, translated_chunk, strict=True)
        ):
            # Validate placeholders against the effective restore input, so plain
            # translated text can still be injected into masked text as before.
            _sanitized, restore_attempt, _should_restore = self._prepare_restore_attempt(
                protected_text,
                translated_protected_text,
            )

            expected_placeholders = self._extract_placeholders_from_text(protected_text.protected)
            actual_placeholders = self._extract_placeholders_from_text(restore_attempt)
            missing_placeholders = sorted(set(expected_placeholders) - set(actual_placeholders))
            unexpected_placeholders = sorted(set(actual_placeholders) - set(expected_placeholders))

            if getattr(self, "_resume_diagnostics_active", False):
                raw_actual_placeholders = self._extract_placeholders_from_text(translated_protected_text)
                trace_missing = self._TRACE_PLACEHOLDER_NAME not in raw_actual_placeholders
                trace_present = self._TRACE_PLACEHOLDER_NAME in raw_actual_placeholders
                if trace_missing or not trace_present:
                    expected_index = protected_text.protected.find(self._TRACE_PLACEHOLDER_NAME)
                    context = ""
                    if translated_protected_text:
                        center = min(max(expected_index, 0), len(translated_protected_text) - 1)
                        start = max(0, center - 75)
                        end = min(len(translated_protected_text), center + 75)
                        context = translated_protected_text[start:end]
                    self.logger.warning(
                        "resume diagnostics: placeholder first missing in translated_protected chunk_index=%s local_index=%s placeholder=%s present=%s missing=%s context=%r",
                        getattr(self, "_resume_diagnostics_chunk_index", "n/a"),
                        chunk_index,
                        self._TRACE_PLACEHOLDER_NAME,
                        trace_present,
                        trace_missing,
                        context,
                    )

            if not missing_placeholders and not unexpected_placeholders:
                continue

            entry_index, entry = matching_entries[0]
            mismatches.append(
                PlaceholderMismatchDetails(
                    chunk_index=chunk_index,
                    entry_index=entry_index,
                    entry=entry,
                    protected_text=protected_text,
                    translated_protected_text=translated_protected_text,
                    expected_placeholders=expected_placeholders,
                    actual_placeholders=actual_placeholders,
                    missing_placeholders=missing_placeholders,
                    unexpected_placeholders=unexpected_placeholders,
                )
            )
        return mismatches

    def _raise_placeholder_mismatch_after_retry(self, mismatches: list[PlaceholderMismatchDetails]) -> Exception:
        debug_dirs: list[str] = []
        for mismatch in mismatches:
            exception_message = (
                "Placeholder mismatch after translation: "
                f"missing={mismatch.missing_placeholders} "
                f"unexpected={mismatch.unexpected_placeholders}"
            )
            debug_dir = self._persist_restore_debug_artifacts(
                exception_message=exception_message,
                original_source=mismatch.protected_text.original,
                protected_source=mismatch.protected_text.protected,
                translated_protected=mismatch.translated_protected_text,
                sanitized_translated=mismatch.translated_protected_text,
                placeholders_before_restore=mismatch.expected_placeholders,
                placeholders_after_restore=mismatch.actual_placeholders,
                file_name=mismatch.entry.file.name,
                field_name=mismatch.entry.field,
                json_path=mismatch.entry.path,
                restored_attempt=mismatch.translated_protected_text,
            )
            debug_dirs.append(str(debug_dir))
            self.logger.error(
                "placeholder mismatch persisted after retry",
                extra={
                    "entry_id": mismatch.entry_index + 1,
                    "field": mismatch.entry.field,
                    "file": str(mismatch.entry.file),
                    "json_path": self._render_json_path(mismatch.entry.path),
                    "missing_placeholders": mismatch.missing_placeholders,
                    "unexpected_placeholders": mismatch.unexpected_placeholders,
                    "debug_dir": str(debug_dir),
                },
            )

        first = mismatches[0]
        return PlaceholderMismatchError(
            "Placeholder mismatch after translation "
            f"for {first.entry.file}:{self._render_json_path(first.entry.path)} "
            f"missing={first.missing_placeholders} unexpected={first.unexpected_placeholders} "
            f"debug_dirs={debug_dirs}"
        )

    def _log_placeholder_trace_stage(
        self,
        *,
        stage: str,
        text: str,
        placeholder_name: str,
        expected_index: int | None,
    ) -> None:
        """Emit presence and local context for a traced placeholder at a pipeline stage."""

        if expected_index is None:
            expected_index = -1

        actual_index = text.find(placeholder_name)
        found = actual_index != -1

        if found:
            center = actual_index
            location = actual_index
        elif expected_index >= 0:
            center = min(expected_index, max(len(text) - 1, 0))
            location = expected_index
        else:
            center = 0
            location = -1

        if text:
            start = max(0, center - 75)
            end = min(len(text), center + 75)
            context = text[start:end]
        else:
            context = ""

        self.logger.info(
            "placeholder trace: stage=%s placeholder=%s present=%s location=%s context=%r",
            stage,
            placeholder_name,
            found,
            location,
            context,
        )

    def _sanitize_translated_protected_text(self, *, translated_text: str, protected_source: str) -> str:
        """Sanitize translated protected text while preserving non-duplicated placeholders."""

        sanitized = self._strip_appended_original_protected_source(
            translated_text=translated_text,
            protected_source=protected_source,
        )

        if translated_text and not sanitized:
            self.logger.warning(
                "sanitization would remove entire translated text; keeping original",
                extra={
                    "translated_length": len(translated_text),
                    "protected_length": len(protected_source),
                },
            )
            return translated_text

        original_placeholders = self._extract_placeholders_from_text(translated_text)
        sanitized_placeholders = self._extract_placeholders_from_text(sanitized)
        original_counter = Counter(original_placeholders)
        sanitized_counter = Counter(sanitized_placeholders)
        removed_counter = original_counter - sanitized_counter

        if removed_counter:
            assert translated_text.startswith(sanitized), (
                "Sanitization removed placeholders from a non-suffix segment"
            )

            removed_tail = translated_text[len(sanitized) :]
            removed_tail_counter = Counter(self._extract_placeholders_from_text(removed_tail))
            assert removed_counter == removed_tail_counter, (
                "Sanitization removed placeholders outside of the stripped suffix"
            )

            removed_tail_is_protected_suffix = bool(removed_tail) and protected_source.endswith(removed_tail)
            if not removed_tail_is_protected_suffix:
                for placeholder_name in removed_counter:
                    assert sanitized_counter.get(placeholder_name, 0) > 0, (
                        "Sanitization removed a non-duplicated placeholder"
                    )

        return sanitized

    def _prepare_restore_attempt(self, protected_text: Any, translated_text: str) -> tuple[str, str, bool]:
        """Return sanitized and restore-attempt texts plus whether restore should run."""

        sanitized_translated_text = self._sanitize_translated_protected_text(
            translated_text=translated_text,
            protected_source=protected_text.protected,
        )

        expected_index = protected_text.protected.find(self._TRACE_PLACEHOLDER_NAME)
        self._log_placeholder_trace_stage(
            stage="after_sanitization",
            text=sanitized_translated_text,
            placeholder_name=self._TRACE_PLACEHOLDER_NAME,
            expected_index=expected_index,
        )

        if not sanitized_translated_text:
            self._log_placeholder_trace_stage(
                stage="after_inject_translation_into_masked_text",
                text=sanitized_translated_text,
                placeholder_name=self._TRACE_PLACEHOLDER_NAME,
                expected_index=expected_index,
            )
            return sanitized_translated_text, sanitized_translated_text, False

        placeholder_pattern = getattr(self.protector, "_PLACEHOLDER_PATTERN", re.compile(r"__FT_[A-Z_]+_\d{5}__"))
        matches = list(placeholder_pattern.finditer(protected_text.protected))
        if not matches:
            self._log_placeholder_trace_stage(
                stage="after_inject_translation_into_masked_text",
                text=sanitized_translated_text,
                placeholder_name=self._TRACE_PLACEHOLDER_NAME,
                expected_index=expected_index,
            )
            return sanitized_translated_text, sanitized_translated_text, False

        translated_placeholders = self._extract_placeholders_from_text(sanitized_translated_text)
        if translated_placeholders:
            self._log_placeholder_trace_stage(
                stage="after_inject_translation_into_masked_text",
                text=sanitized_translated_text,
                placeholder_name=self._TRACE_PLACEHOLDER_NAME,
                expected_index=expected_index,
            )
            return sanitized_translated_text, sanitized_translated_text, True

        translated_masked_text = self._inject_translation_into_masked_text(
            protected_text.protected,
            sanitized_translated_text,
        )
        self._log_placeholder_trace_stage(
            stage="after_inject_translation_into_masked_text",
            text=translated_masked_text,
            placeholder_name=self._TRACE_PLACEHOLDER_NAME,
            expected_index=expected_index,
        )
        return sanitized_translated_text, translated_masked_text, True

    def _restore_protected_text(self, protected_text: Any, translated_text: str) -> str:
        """Restore protected markers into translated text while preserving surrounding content."""

        expected_index = protected_text.protected.find(self._TRACE_PLACEHOLDER_NAME)
        self._log_placeholder_trace_stage(
            stage="protected_source",
            text=protected_text.protected,
            placeholder_name=self._TRACE_PLACEHOLDER_NAME,
            expected_index=expected_index,
        )
        self._log_placeholder_trace_stage(
            stage="translated_protected",
            text=translated_text,
            placeholder_name=self._TRACE_PLACEHOLDER_NAME,
            expected_index=expected_index,
        )

        _, restored_attempt, should_restore = self._prepare_restore_attempt(
            protected_text, translated_text
        )

        if not should_restore:
            return restored_attempt

        self._log_placeholder_trace_stage(
            stage="before_protect_restore",
            text=restored_attempt,
            placeholder_name=self._TRACE_PLACEHOLDER_NAME,
            expected_index=expected_index,
        )

        expected = set(protected_text.placeholders)
        actual = set(self._extract_placeholders_from_text(restored_attempt))
        missing = expected - actual
        unexpected = actual - expected
        restore_input_hash = hashlib.sha256(restored_attempt.encode("utf-8")).hexdigest()[:16]

        self.logger.info(
            "restore precheck (translator): len=%s sha256_16=%s expected=%s actual=%s missing=%s unexpected=%s",
            len(restored_attempt),
            restore_input_hash,
            sorted(expected),
            sorted(actual),
            sorted(missing),
            sorted(unexpected),
        )

        return self.protector.restore(protected_text, restored_attempt)

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
        return self._build_prompt_from_request_items(
            self._build_request_items(texts),
            source_language=source_language,
            target_language=target_language,
            glossary=glossary,
        )

    def _build_prompt_from_request_items(
        self,
        request_items: list[dict[str, Any]],
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
        lines.append(json.dumps(request_items, ensure_ascii=False, indent=2))

        return "\n".join(lines)

    def _parse_response_translations_by_id(
        self,
        response_text: str,
        *,
        requested_ids: list[int],
    ) -> tuple[dict[int, str], list[int]]:
        if not response_text:
            return {}, requested_ids

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

        return translations_by_id, missing_ids

    def _parse_response(self, response_text: str, *, requested_ids: list[int]) -> list[str]:
        translations_by_id, missing_ids = self._parse_response_translations_by_id(
            response_text,
            requested_ids=requested_ids,
        )
        if missing_ids:
            raise OpenAITranslatorCountError(f"Missing translation ids in response: {missing_ids}")

        return [translations_by_id[identifier] for identifier in requested_ids]
