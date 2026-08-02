"""Protection and restoration of Foundry markup before and after translation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


@dataclass(slots=True)
class ProtectedText:
    """Protected text with placeholder-based masking for Foundry-only syntax."""

    original: str
    protected: str
    placeholders: dict[str, str]


class MarkupProtector:
    """Protect Foundry markup before translation and restore it afterward.

    The flow is intentionally simple:
    1. protect() replaces Foundry-specific syntax with stable placeholders;
    2. the translation model sees only the masked text;
    3. restore() injects the translated visible text back into the original string
       and then restores the original protected markup exactly.
    """

    _PLACEHOLDER_PREFIX: Final[str] = "__FOUNDRY_PLACEHOLDER_"

    def protect(self, text: str) -> ProtectedText:
        """Return a masked copy of the text while keeping the original untouched."""

        placeholders: dict[str, str] = {}
        protected = text

        for pattern in (
            r"@Embed\[[^\]]*\]",
            r"@UUID\[[^\]]*\]",
            r"@Check\[[^\]]*\]",
            r"@Damage\[[^\]]*\]",
            r"@Template\[[^\]]*\]",
            r"\[\[[^\]]+\]\]",
            r"\{\{[^{}]+\}\}",
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            r"<[^>]+>",
            r'\b([a-zA-Z_:][-a-zA-Z0-9_:.]*)="[^"]*"',
        ):
            protected = self._replace_matches(protected, pattern, placeholders)

        return ProtectedText(
            original=text,
            protected=protected,
            placeholders=placeholders,
        )

    def restore(self, protected_text: ProtectedText, translated_text: str) -> str:
        """Replace each placeholder in the translated masked text with its original value."""

        if not translated_text:
            return protected_text.original

        if not protected_text.placeholders:
            return translated_text

        return re.sub(
            r"__FOUNDRY_PLACEHOLDER_[0-9]+__",
            lambda match: protected_text.placeholders.get(match.group(0), match.group(0)),
            translated_text,
        )

    def _replace_matches(self, text: str, pattern: str, placeholders: dict[str, str]) -> str:
        def repl(match: re.Match[str]) -> str:
            original = match.group(0)
            placeholder = f"{self._PLACEHOLDER_PREFIX}{len(placeholders)}__"
            placeholders[placeholder] = original
            return placeholder

        return re.sub(pattern, repl, text)
