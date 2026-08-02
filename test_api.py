from foundry_translator.openai_translator import OpenAITranslator

translator = OpenAITranslator(batch_size=5)

texts = [
    "Bandit Ambush",
    "Goblin",
    "Longsword",
    "Shortbow",
    "Treasure Chest",
]

result = translator.translate_batch(
    texts,
    source_language="English",
    target_language="French",
)

for r in result:
    print(r)