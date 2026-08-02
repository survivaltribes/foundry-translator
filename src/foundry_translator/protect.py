"""Protect and restore Foundry-specific syntax before and after translation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


@dataclass(slots=True)
class Placeholder:
    """Metadata describing a protected placeholder."""

    category: str
    index: int
    original: str


@dataclass(slots=True)
class ProtectedText:
    """Masked text and the mapping needed to restore protected tokens."""

    original: str
    protected: str
    placeholders: dict[str, Placeholder | str]


class Protect:
    """Protect Foundry-specific tokens before sending text to a translator."""

    _PLACEHOLDER_PREFIX: Final[str] = "__FT_"
    _PLACEHOLDER_PATTERN: Final[re.Pattern[str]] = re.compile(r"__FT_[A-Z_]+_\d{5}__")

    _UUID_PATTERN: Final[re.Pattern[str]] = re.compile(r"@UUID\[[^\]]*\]")
    _EMBED_PATTERN: Final[re.Pattern[str]] = re.compile(r"@Embed\[[^\]]*\]")
    _CHECK_PATTERN: Final[re.Pattern[str]] = re.compile(r"@Check\[[^\]]*\]")
    _DAMAGE_PATTERN: Final[re.Pattern[str]] = re.compile(r"@Damage\[[^\]]*\]")
    _TEMPLATE_PATTERN: Final[re.Pattern[str]] = re.compile(r"@Template\[[^\]]*\]")
    _REFERENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"&Reference\[[^\]]*\]")
    _DOUBLE_BRACKET_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[\[[^\]]+\]\]")
    _MACRO_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{\{[^{}]+\}\}")
    _UUID_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    )
    _MARKDOWN_LINK_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[[^\]]+\]\([^\)]+\)")
    _HTML_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
    _ATTRIBUTE_PATTERN: Final[re.Pattern[str]] = re.compile(r'\b([a-zA-Z_:][-a-zA-Z0-9_:.]*)="[^"]*"')
    _ROLL_COMMAND_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"\b(?:\d+d\d+|\d+d\d+[+-]?\d*|@[a-zA-Z]+\[[^\]]+\])\b"
    )
    _IMAGE_REFERENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"!\[[^\]]*\]\([^\)]+\)")

    _PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
        ("UUID", _UUID_PATTERN),
        ("EMBED", _EMBED_PATTERN),
        ("CHECK", _CHECK_PATTERN),
        ("DAMAGE", _DAMAGE_PATTERN),
        ("TEMPLATE", _TEMPLATE_PATTERN),
        ("REFERENCE", _REFERENCE_PATTERN),
        ("DOUBLE_BRACKET", _DOUBLE_BRACKET_PATTERN),
        ("MACRO", _MACRO_PATTERN),
        ("UUID_TOKEN", _UUID_TOKEN_PATTERN),
        ("MARKDOWN_LINK", _MARKDOWN_LINK_PATTERN),
        ("HTML", _HTML_TAG_PATTERN),
        ("ATTRIBUTE", _ATTRIBUTE_PATTERN),
        ("ROLL_COMMAND", _ROLL_COMMAND_PATTERN),
        ("IMAGE", _IMAGE_REFERENCE_PATTERN),
    )

    def protect(self, text: str) -> ProtectedText:
        """Return a protected version of the text with Foundry tokens masked."""

        if not text:
            return ProtectedText(original=text, protected=text, placeholders={})

        placeholders: dict[str, Placeholder | str] = {}
        counts: dict[str, int] = {category: 0 for category, _pattern in self._PATTERNS}
        result: list[str] = []
        cursor = 0

        while cursor < len(text):
            best_match: tuple[int, int, str] | None = None

            for category, pattern in self._PATTERNS:
                match = pattern.search(text, cursor)
                if match is None:
                    continue

                start, end = match.span()
                if best_match is None or start < best_match[0] or (
                    start == best_match[0] and end > best_match[1]
                ):
                    best_match = (start, end, category)

            if best_match is None:
                result.append(text[cursor:])
                break

            start, end, category = best_match
            if start > cursor:
                result.append(text[cursor:start])

            counts[category] += 1
            placeholder = self._make_placeholder(category, counts[category])
            placeholder_value = text[start:end]
            placeholders[placeholder] = Placeholder(
                category=category,
                index=counts[category],
                original=placeholder_value,
            )
            result.append(placeholder)
            cursor = end

        protected = "".join(result)
        return ProtectedText(original=text, protected=protected, placeholders=placeholders)

    def restore(
        self,
        protected_text: ProtectedText,
        translated_text: str | None = None,
    ) -> str:
        """Restore protected tokens into the translated masked text."""

        if not protected_text.placeholders:
            return protected_text.protected if translated_text is None else translated_text

        text_to_restore = protected_text.protected if translated_text is None else translated_text
        placeholders_in_text = self._find_placeholders(text_to_restore)
        if self._has_duplicate_placeholders(placeholders_in_text):
            raise ValueError("Duplicate placeholders detected in protected text")

        for placeholder_name in placeholders_in_text:
            if not self._is_valid_placeholder_name(placeholder_name):
                raise ValueError(f"Malformed placeholder encountered: {placeholder_name}")

        for placeholder_name in protected_text.placeholders:
            if not self._is_valid_placeholder_name(placeholder_name):
                raise ValueError(f"Malformed placeholder encountered: {placeholder_name}")

        placeholder_names = set(protected_text.placeholders)
        placeholder_names_in_text = set(placeholders_in_text)

        missing = placeholder_names_in_text - placeholder_names
        if missing:
            raise ValueError(f"Missing placeholders: {sorted(missing)}")

        extra = placeholder_names - placeholder_names_in_text
        if extra:
            raise ValueError(f"Unexpected placeholders not present in protected text: {sorted(extra)}")

        restored = text_to_restore
        for placeholder_name in placeholders_in_text:
            placeholder = protected_text.placeholders.get(placeholder_name)
            if placeholder is None:
                raise ValueError(f"Placeholder not found in mapping: {placeholder_name}")
            replacement = placeholder.original if isinstance(placeholder, Placeholder) else str(placeholder)
            restored = restored.replace(placeholder_name, replacement, 1)
        return restored

    def _find_placeholders(self, text: str) -> list[str]:
        """Return placeholder names found in the provided text in order."""

        return [match.group(0) for match in self._PLACEHOLDER_PATTERN.finditer(text)]

    def _has_duplicate_placeholders(self, placeholder_names: list[str]) -> bool:
        """Return True when the same placeholder appears more than once."""

        return len(set(placeholder_names)) != len(placeholder_names)

    def _make_placeholder(self, category: str, index: int) -> str:
        """Create a deterministic placeholder for a protected token."""

        return f"{self._PLACEHOLDER_PREFIX}{category.upper()}_{index:05d}__"

    def _is_valid_placeholder_name(self, placeholder_name: str) -> bool:
        """Return True when the placeholder name uses the expected format."""

        return bool(re.fullmatch(r"__FT_[A-Z_]+_\d{5}__", placeholder_name))


class MarkupProtector(Protect):
    """Backward-compatible name for the shared protection implementation."""


def protect(text: str) -> ProtectedText:
    """Convenience function for protecting text."""

    return Protect().protect(text)


def restore(protected_text: ProtectedText, translated_text: str | None = None) -> str:
    """Convenience function for restoring protected text."""

    return Protect().restore(protected_text, translated_text)
