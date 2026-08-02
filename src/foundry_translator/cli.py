"""Command-line interface for the Foundry Translator pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import Pipeline
from .translator import DummyTranslator
from .writer import JsonWriter

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m foundry_translator")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Translate a source directory")
    run_parser.add_argument("--input", required=True, help="Directory containing the source JSON files")
    run_parser.add_argument("--output", required=True, help="Directory where translated files will be written")

    validate_parser = subparsers.add_parser("validate", help="Validate translated files in a directory")
    validate_parser.add_argument("output_directory", help="Directory containing generated translated JSON files")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return validate_output_dir(Path(args.output_directory).expanduser().resolve())

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return 2

    pipeline = Pipeline(translator=DummyTranslator())
    result = pipeline.run(input_dir, output_dir)

    logger.info("Files analyzed: %s", result.scanned_files)
    logger.info("Texts translated: %s", result.translated_entries)
    logger.info("Duration: %.2fs", result.duration_seconds)
    logger.info("Errors: %s", result.errors)
    return 0 if result.errors == 0 else 1


def validate_output_dir(output_dir: Path) -> int:
    if not output_dir.exists():
        logger.error("Output directory does not exist: %s", output_dir)
        return 2

    writer = JsonWriter()
    files = sorted(output_dir.rglob("*.json"))
    if not files:
        logger.error("No JSON files found in %s", output_dir)
        return 1

    for file in files:
        try:
            payload = writer.load(file)
            writer._validate_payload(payload)
        except Exception as exc:  # pragma: no cover - CLI safety branch
            logger.error("Validation failed for %s: %s", file, exc)
            return 1

    logger.info("Validation passed for %s files", len(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
