from __future__ import annotations

import pytest

from foundry_translator.protect import Protect, ProtectedText


def test_protect_and_restore_round_trip_foundry_syntax() -> None:
    protector = Protect()
    original = (
        "Use @UUID[123e4567-e89b-12d3-a456-426614174000] and @Embed[actor] "
        "before [[macro]] and ![img](path/to/image.png) and <b>bold</b> "
        "with 3d6 and [link](https://example.com)."
    )

    protected = protector.protect(original)

    assert protected.original == original
    assert protected.protected != original
    assert "@UUID[" not in protected.protected
    assert "[[macro]]" not in protected.protected
    assert "![img](path/to/image.png)" not in protected.protected
    assert "<b>" not in protected.protected
    assert "3d6" not in protected.protected
    assert "[link](https://example.com)" not in protected.protected

    restored = protector.restore(protected)
    assert restored == original


def test_protect_and_restore_square_bracket_references() -> None:
    protector = Protect()
    original = "Use [Fire Damage] against creatures rated [CR 5]."

    protected = protector.protect(original)

    assert "[Fire Damage]" not in protected.protected
    assert "[CR 5]" not in protected.protected
    translated = protected.protected.replace("Use", "Utilisez").replace(
        "against creatures rated", "contre les creatures de niveau"
    )
    assert protector.restore(protected, translated) == (
        "Utilisez [Fire Damage] contre les creatures de niveau [CR 5]."
    )


def test_restore_rejects_malformed_placeholder_identifiers() -> None:
    protector = Protect()
    malformed = ProtectedText(
        original="Hello",
        protected="__FT_UUID_1__",
        placeholders={"__FT_UUID_1__": "placeholder"},
    )

    with pytest.raises(ValueError, match="Malformed placeholder"):
        protector.restore(malformed)


def test_restore_rejects_duplicate_placeholders_in_text() -> None:
    protector = Protect()
    duplicate = ProtectedText(
        original="Hello",
        protected="__FT_UUID_00001____FT_UUID_00001__",
        placeholders={
            "__FT_UUID_00001__": "value",
        },
    )

    with pytest.raises(ValueError, match="Duplicate placeholders"):
        protector.restore(duplicate)
