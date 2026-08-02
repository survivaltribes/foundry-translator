from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from foundry_translator.cli import validate_output_dir
from foundry_translator.openai_translator import OpenAITranslator
from foundry_translator.pipeline import Pipeline
from foundry_translator.translation_cache import TranslationCache
from foundry_translator.writer import JsonWriter


class MockOpenAITranslator(OpenAITranslator):
    def __init__(self, *, cache: TranslationCache | None = None) -> None:
        self.client = Mock()
        super().__init__(
            api_key="test-key",
            model="gpt-4.1-mini",
            target_language="French",
            batch_size=2,
            client=self.client,
            cache=cache,
        )

    def _call_openai(self, prompt: str) -> str:
        entries = []
        for line in prompt.splitlines():
            if not line.strip():
                continue
            match = re.match(r"^\d+\.\s*(.+)$", line)
            if match:
                entries.append(match.group(1))

        translations = [self._translate_entry(entry) for entry in entries]
        return "\n".join(translations)

    def _translate_entry(self, text: str) -> str:
        if "Welcome" in text or "Bienvenue" in text:
            return "Bienvenue"
        return "Bonjour"


def test_full_foundry_integration(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures/foundry_module")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)

    source_file = input_dir / "sample.json"
    source_file.write_text((fixture_dir / "sample.json").read_text(encoding="utf-8"), encoding="utf-8")

    cache = TranslationCache(tmp_path / "translation_cache.json")
    translator = MockOpenAITranslator(cache=cache)
    pipeline = Pipeline(translator=translator, writer=JsonWriter(), cache=cache)

    result = pipeline.run(input_dir, output_dir)

    generated = (output_dir / "sample.translated.json").read_text(encoding="utf-8")
    expected = (fixture_dir / "expected.json").read_text(encoding="utf-8")

    assert result.translated_entries == 4
    assert result.scanned_files == 1

    generated_payload = json.loads(generated)
    expected_payload = json.loads(expected)

    assert generated_payload == expected_payload
    assert list(generated_payload.keys()) == list(expected_payload.keys())
    assert generated_payload["name"] == expected_payload["name"]
    assert generated_payload["description"] == expected_payload["description"]
    assert generated_payload["items"][0]["uuid"] == expected_payload["items"][0]["uuid"]
    assert generated_payload["items"][0]["text"] == expected_payload["items"][0]["text"]
    assert "__FOUNDRY_PLACEHOLDER_" not in json.dumps(generated_payload)
    assert json.loads(generated) == expected_payload
    assert validate_output_dir(output_dir) == 0
