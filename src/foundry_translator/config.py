"""
Configuration du projet Foundry Translator.
Charge les variables d'environnement (.env) et fournit
une configuration centralisée.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Charge automatiquement le fichier .env s'il existe
load_dotenv()


@dataclass(slots=True)
class Settings:
    """Configuration globale."""

    api_key: str
    model: str
    batch_size: int
    source_language: str
    target_language: str
    glossary_path: Path
    cache_enabled: bool


def load_settings() -> Settings:
    """
    Charge la configuration depuis les variables d'environnement.
    """

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY est absente.\n"
            "Crée un fichier .env à la racine du projet."
        )

    return Settings(
        api_key=api_key,
        model=os.getenv("MODEL", "gpt-5.5"),
        batch_size=int(os.getenv("BATCH_SIZE", "100")),
        source_language=os.getenv("SOURCE_LANGUAGE", "en"),
        target_language=os.getenv("TARGET_LANGUAGE", "fr"),
        glossary_path=Path(
            os.getenv(
                "GLOSSARY_PATH",
                "glossary/dnd2024_fr.json"
            )
        ),
        cache_enabled=os.getenv(
            "CACHE_ENABLED",
            "true"
        ).lower() == "true",
    )