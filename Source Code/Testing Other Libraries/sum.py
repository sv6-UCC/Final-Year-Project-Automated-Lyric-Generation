
from transformers import GPT2Tokenizer, GPT2LMHeadModel


model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_prompt = "Write a sentence that contains the word 'car':"
input_encoding = tokenizer.encode(prompt, return_tensors="pt")
new_sentence = model.generate(input_encoding, max_length=10, temperature=0.6)


output = tokenizer.decode(new_sentence[0], skip_special_tokens=True)
print(output)


from aitextgen import aitextgen

ai = aitextgen()

ai.generate(n=3, prompt="I believe in unicorns because", max_length=20)
ai.generate_to_file(n=10, prompt="I believe in unicorns because", max_length=20, temperature=1.2)
