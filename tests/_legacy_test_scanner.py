from pathlib import Path
import sys

# Permet d'importer le package depuis le dossier src
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from foundry_translator.scanner import Scanner


def main():

    print("=" * 60)
    print("Foundry Translator - Scanner")
    print("=" * 60)

    folder = Path(
        input("\nChemin du dossier Babele : ").strip()
    )

    if not folder.exists():
        print("\n❌ Dossier introuvable.")
        return

    scanner = Scanner(folder)

    documents, entries = scanner.scan()

    print(f"\n{len(documents)} fichier(s) trouvé(s)")
    print(f"{len(entries)} texte(s) détecté(s)\n")

    for entry in entries[:20]:
        print("-" * 60)
        print(entry.file.name)
        print(entry.path)
        print(entry.source[:120])

    print("\nScan terminé.")


if __name__ == "__main__":
    main()