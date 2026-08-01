# Foundry Translator

Translate Babele JSON files for Foundry VTT using OpenAI.

## Features

- Detect Babele JSON files
- Translate only translatable fields
- Preserve UUIDs
- Preserve HTML
- Preserve Foundry links
- D&D glossary
- Translation cache
- Batch translation
- Resume after interruption

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
pip install -r requirements.txt
```

## Usage

```bash
python translate.py "C:\Foundry\modules\my-module\fr\compendium-export"
```

## Status

🚧 In development
