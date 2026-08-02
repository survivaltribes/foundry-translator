# Foundry Translator

Translate Babele JSON files for Foundry VTT using OpenAI.

## Features

- Recursively scan exported JSON files for translatable content
- Detect only supported translatable fields
- Return structured `TranslationEntry` objects with file, JSON path, field name, and source text
- Leave the JSON files unchanged

## Supported fields

- name
- description
- caption
- content
- text
- label
- hint

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Scan a folder

```bash
python3 test_scanner.py /path/to/babele-export
```

If no path is provided, the script prompts for one.

### Python API

```python
from pathlib import Path
from foundry_translator.scanner import Scanner

scanner = Scanner(Path("/path/to/babele-export"))
documents, entries = scanner.scan()

for entry in entries:
    print(entry.file, entry.path, entry.field, entry.source)
```

## Status

🚧 In development
