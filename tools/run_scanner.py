#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from foundry_translator.scanner import Scanner


def main() -> int:
    print("=" * 60)
    print("Foundry Translator - Scanner")
    print("=" * 60)

    if len(sys.argv) > 1:
        folder = Path(sys.argv[1]).expanduser()
    else:
        folder = Path(input("\nChemin du dossier Babele : ").strip())

    if not folder.exists():
        print("\n❌ Dossier introuvable.")
        return 1

    scanner = Scanner(folder)
    documents, entries, issues = scanner.scan()

    print(f"\n{len(documents)} fichier(s) trouvé(s)")
    print(f"{len(entries)} texte(s) détecté(s)\n")

    if issues:
        print(f"{len(issues)} problème(s) non bloquant(s) détecté(s) :")
        for issue in issues:
            print(f"- {issue.file}: {issue.message}")
        print()

    for entry in entries[:20]:
        print("-" * 60)
        print(entry.file.name)
        print(entry.path)
        print(entry.source[:120])

    print("\nScan terminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
