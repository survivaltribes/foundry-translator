"""OpenAI-backed text translator using the Responses API."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import replace
from typing import Any

from openai import APIConnectionError
from openai import APITimeoutError
from openai import APIStatusError
from openai import OpenAI

from .markup import MarkupProtector
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
        max_retries: int = 3,
        timeout: float = 60.0,
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
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.temperature = temperature
        self.top_p = top_p
        self.client = client or OpenAI(api_key=self.api_key, timeout=self.timeout)
        self.logger = logger or logging.getLogger(__name__)
        self.cache = cache
        self.protector = MarkupProtector()

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

        translated: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            translated.extend(
                self._translate_batch(
                    batch,
                    source_language=source_language,
                    target_language=target_language,
                    glossary=glossary,
                )
            )

        if len(translated) != len(texts):
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

        protected_texts = [protected_text for _, _, protected_text in pending_groups]
        translated_protected_texts = self._translate_batch(
            [protected.protected for protected in protected_texts],
            source_language="English",
            target_language=self.target_language,
        )

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

        prompt = self._build_prompt(texts, source_language=source_language, target_language=target_language, glossary=glossary)
        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.info(
                    "requesting translations",
                    extra={
                        "batch_size": len(texts),
                        "attempt": attempt,
                        "model": self.model,
                        "source_language": source_language,
                        "target_language": target_language,
                    },
                )
                response_text = self._call_openai(prompt)
                translations = self._parse_response(response_text, expected_count=len(texts))
                self.logger.info(
                    "translations completed",
                    extra={
                        "batch_size": len(texts),
                        "attempt": attempt,
                        "model": self.model,
                    },
                )
                return translations
            except OpenAITranslatorCountError as exc:
                if attempt >= self.max_retries:
                    raise
                delay = self.retry_delay * (self.retry_backoff_factor ** (attempt - 1))
                self.logger.warning(
                    "translation count mismatch; retrying",
                    extra={
                        "batch_size": len(texts),
                        "attempt": attempt,
                        "delay_seconds": delay,
                        "error": str(exc),
                    },
                )
                time.sleep(delay)
            except (APITimeoutError, APIConnectionError, APIStatusError) as exc:  # type: ignore[misc]
                if attempt >= self.max_retries:
                    raise OpenAITranslatorTimeoutError(
                        f"OpenAI request failed after {self.max_retries} attempts"
                    ) from exc
                delay = self.retry_delay * (self.retry_backoff_factor ** (attempt - 1))
                self.logger.warning(
                    "translation request failed; retrying",
                    extra={
                        "batch_size": len(texts),
                        "attempt": attempt,
                        "delay_seconds": delay,
                        "error": str(exc),
                    },
                )
                time.sleep(delay)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise OpenAITranslatorRequestError("OpenAI request failed") from exc

        raise OpenAITranslatorRequestError("OpenAI request failed")

    def _call_openai(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
        )
        return getattr(response, "output_text", "")

    def _restore_protected_text(self, protected_text: Any, translated_text: str) -> str:
        if not translated_text:
            return translated_text

        matches = list(re.finditer(r"__FOUNDRY_PLACEHOLDER_[0-9]+__", protected_text.protected))
        if not matches:
            return translated_text

        translated_masked_text = self._inject_translation_into_masked_text(protected_text.protected, translated_text)
        return self.protector.restore(protected_text, translated_masked_text)

    def _inject_translation_into_masked_text(self, masked_text: str, translated_text: str) -> str:
        matches = list(re.finditer(r"__FOUNDRY_PLACEHOLDER_[0-9]+__", masked_text))
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

    def _contains_placeholder(self, text: str) -> bool:
        return "__FOUNDRY_PLACEHOLDER_" in text

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
