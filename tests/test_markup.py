from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from foundry_translator.markup import MarkupProtector
from foundry_translator.openai_translator import OpenAITranslator
from foundry_translator.scanner import TranslationEntry


def test_markup_protector_restores_original_text_identically() -> None:
    protector = MarkupProtector()
    original = (
        '<p class="hero" data-id="42">Hello @Embed[foo] and {{macro}} and [[link]] '
        '@UUID[123e4567-e89b-12d3-a456-426614174000] @Check[skill] @Damage[1d6] '
        '@Template[spell] and 123e4567-e89b-12d3-a456-426614174000</p>'
    )

    protected = protector.protect(original)

    assert protected.original == original
    assert protected.protected != original
    assert '@Embed[' not in protected.protected
    assert '@UUID[' not in protected.protected
    assert '@Check[' not in protected.protected
    assert '@Damage[' not in protected.protected
    assert '@Template[' not in protected.protected
    assert '[[link]]' not in protected.protected
    assert '{{macro}}' not in protected.protected
    assert '<p' not in protected.protected

    restored = protector.restore(protected, protected.protected.replace('Hello', 'Bonjour'))
    assert restored == original.replace('Hello', 'Bonjour')


def test_restore_removes_all_placeholders() -> None:
    protector = MarkupProtector()
    original = '<p class="hero">Hello @Embed[foo] {{macro}}</p>'
    protected = protector.protect(original)

    restored = protector.restore(protected, protected.protected.replace("Hello", "Bonjour"))

    assert "__FOUNDRY_PLACEHOLDER_" not in restored
    assert restored == '<p class="hero">Bonjour @Embed[foo] {{macro}}</p>'


def test_openai_translator_restores_markup_after_translation() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"translations": [{"id": 1, "translation": "Bonjour"}]}'
    )

    translator = OpenAITranslator(
        api_key="test-key",
        model="gpt-4.1-mini",
        target_language="French",
        batch_size=1,
        client=client,
    )

    entry = TranslationEntry(
        file=Path("sample.json"),
        path=["name"],
        field="name",
        source='<p class="hero">Hello @Embed[foo] {{macro}}</p>',
    )

    translated = translator.translate([entry])

    assert translated[0].source == '<p class="hero">Bonjour @Embed[foo] {{macro}}</p>'
