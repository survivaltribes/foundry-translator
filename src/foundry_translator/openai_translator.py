"""OpenAI-backed text translator using the Responses API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import time
from dataclasses import replace
from typing import Any

from openai import APIConnectionError
from openai import APITimeoutError
from openai import APIStatusError
from openai import OpenAI

from .protect import Protect
from .scanner import TranslationEntry


class OpenAITranslatorError(RuntimeError):
    """Base exception for translator failures."""


class OpenAITranslatorRequestError(OpenAITranslatorError):
    """Raised when the OpenAI request fails."""


class OpenAITranslatorTimeoutError(OpenAITranslatorRequestError):
    """Raised when the OpenAI request exceeds the configured timeout."""


class OpenAITranslatorCountError(OpenAITranslatorError):
    """Raised when the API returns an unexpected number of translations."""


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
            restored_source = self._restore_protected_text(protected_text, translated_protected_text)
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

        if len(texts) == 1:
            prompt = self._build_prompt(
                texts,
                source_language=source_language,
                target_language=target_language,
                glossary=glossary,
            )
            self._log_batch_start(texts, batch_number=batch_number, total_batches=total_batches, prompt=prompt)
            response_text: str | None = None
            try:
                response_text = self._call_openai(prompt)
                self._last_prompt_for_count_error = prompt
                self._last_response_for_count_error = response_text
                translations = self._parse_response(response_text, expected_count=len(texts))
                self._log_batch_success(texts, batch_number=batch_number, total_batches=total_batches)
                return translations
            except (APITimeoutError, APIConnectionError, APIStatusError) as exc:  # type: ignore[misc]
                self._log_openai_exception(exc, batch_number=batch_number, total_batches=total_batches)
                raise
            except OpenAITranslatorCountError as exc:
                expected_count, received_count = self._extract_translation_counts(exc)
                prompt_path, response_path = self._persist_failed_count_debug_artifacts(
                    prompt=prompt,
                    response_text=response_text or "",
                )
                self.logger.warning(
                    "translation count mismatch for final batch",
                    extra={
                        "expected_translation_count": expected_count,
                        "received_translation_count": received_count,
                        "batch_number": batch_number,
                        "prompt_length": len(prompt),
                        "prompt_path": str(prompt_path),
                        "response_path": str(response_path),
                        "total_batches": total_batches,
                        "batch_size": len(texts),
                        "error": str(exc),
                    },
                )
                raise
            except Exception as exc:  # pragma: no cover - defensive guard
                self._log_openai_exception(exc, batch_number=batch_number, total_batches=total_batches)
                if isinstance(exc, OpenAITranslatorRequestError):
                    raise
                raise OpenAITranslatorRequestError("OpenAI request failed") from exc

        prompt = self._build_prompt(
            texts,
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
                translations = self._parse_response(response_text, expected_count=len(texts))
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
                if attempt >= self.max_retries:
                    raise
                delay = self.retry_delay * (self.retry_backoff_factor ** (attempt - 1))
                self.logger.warning(
                    "translation count mismatch; retrying",
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
        }
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout

        batch_size = len([line for line in prompt.splitlines() if line[:1].isdigit()])
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

    def _get_debug_artifact_paths(self) -> tuple[Path, Path]:
        project_root = Path(__file__).resolve().parents[2]
        debug_dir = project_root / "debug"
        return debug_dir / "failed_prompt.txt", debug_dir / "failed_response.txt"

    def _persist_failed_count_debug_artifacts(self, *, prompt: str, response_text: str) -> tuple[Path, Path]:
        prompt_path, response_path = self._get_debug_artifact_paths()
        try:
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")
            response_path.write_text(response_text, encoding="utf-8")
            self.logger.info(
                "saved OpenAI count mismatch debug artifacts",
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

    def _restore_protected_text(self, protected_text: Any, translated_text: str) -> str:
        """Restore protected markers into translated text while preserving surrounding content."""

        if not translated_text:
            return translated_text

        placeholder_pattern = getattr(self.protector, "_PLACEHOLDER_PATTERN", re.compile(r"__FT_[A-Z_]+_\d{5}__"))
        matches = list(placeholder_pattern.finditer(protected_text.protected))
        if not matches:
            return translated_text

        translated_masked_text = self._inject_translation_into_masked_text(protected_text.protected, translated_text)
        return self.protector.restore(protected_text, translated_masked_text)

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
            "Return exactly one translated line per input line, preserving the same order.",
            "Do not add commentary, numbering, bullets, or extra prose.",
            "Keep placeholders, markup, and code-like tokens unchanged.",
            "Use deterministic, concise, natural phrasing.",
        ]

        if glossary:
            instructions.append("Glossary:")
            for term, translation in sorted(glossary.items()):
                instructions.append(f"{term} -> {translation}")

        lines = ["\n".join(instructions), "", "Inputs:"]
        for index, text in enumerate(texts, start=1):
            lines.append(f"{index}. {text}")

        return "\n".join(lines)

    def _parse_response(self, response_text: str, *, expected_count: int) -> list[str]:
        if not response_text:
            raise OpenAITranslatorCountError(f"Expected {expected_count} translations but received 0")

        text = response_text.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise OpenAITranslatorCountError("Response was not valid JSON") from exc
            if not isinstance(parsed, list):
                raise OpenAITranslatorCountError("Expected a JSON array of translations")
            if len(parsed) != expected_count:
                raise OpenAITranslatorCountError(
                    f"Expected {expected_count} translations but received {len(parsed)}"
                )
            return [str(item).strip() for item in parsed]

        translations = [line.strip() for line in text.splitlines() if line.strip()]
        if len(translations) != expected_count:
            raise OpenAITranslatorCountError(
                f"Expected {expected_count} translations but received {len(translations)}"
            )
        return translations
