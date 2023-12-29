from transformers import pipeline, set_seed


model = pipeline("text-generation", model="bert-base-uncased")


set_seed(42)


seed = "Can anybody"


generated_lyrics = model(seed, max_length=10, num_return_sequences=1)


for lyrics in generated_lyrics:
    print(lyrics['generated_text'])
