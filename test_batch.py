from pathlib import Path

from foundry_translator.scanner import Scanner
from foundry_translator.openai_translator import OpenAITranslator

folder = Path("/mnt/c/Users/survi/AppData/Local/FoundryVTT/Data/modules/dnd-heroes-borderlands/fr/compendium-export")

scanner = Scanner(folder)

_, entries, _ = scanner.scan()

translator = OpenAITranslator(batch_size=20)

texts = [e.source for e in entries[:20]]

result = translator.translate_batch(
    texts,
    source_language="English",
    target_language="French",
)

print("Nombre de traductions :", len(result))
for t in result[:5]:
    print(t)