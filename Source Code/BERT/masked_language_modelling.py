from transformers import AutoTokenizer
from transformers import TFAutoModelForMaskedLM
import tensorflow as tf
import numpy as np

grace_kelly_prediction = "I could be [MASK]."

masked_model = TFAutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

masked_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

inputs_to_model = masked_tokenizer(grace_kelly_prediction, return_tensors="np")
model_logits = masked_model(**inputs_to_model).logits
tokenizer_index = np.argwhere(inputs_to_model["input_ids"] == masked_tokenizer.mask_token_id)[0, 1]
tokenizer_logits = model_logits[0, tokenizer_index, :]
five_predictions = np.argsort(-tokenizer_logits)[:5].tolist()

for lyric in five_predictions:
    print(f">>> {grace_kelly_prediction.replace(masked_tokenizer.mask_token, masked_tokenizer.decode([lyric]))}")

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
mylist=["there","no","room"]
for i in range(0,len(mylist)):

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    text = "the heart is a bloom, there no room"
    target_mask = mylist[i]
    tokenized_text = tokenizer.tokenize(text)

    
    masked_index = tokenized_text.index(target_mask)
    tokenized_text[masked_index] = '[MASK]'

    
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    
    segments_ids = [1] * len(tokenized_text)
    
    segments_ids[0] = 0
    segments_ids[1] = 0

    
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.eval()

    predictions = model(tokens_tensor, segments_tensors)
    new_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_word = tokenizer.convert_ids_to_tokens([new_index])

    print("Original sentence:", text)
    print("Masked sentence:", " ".join(tokenized_text))

    print("Predicted word:", predicted_word)
    print("Other words:")

    for a in range(10):
        predictions[0,masked_index,new_index] = -11100000
        new_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_word = tokenizer.convert_ids_to_tokens([new_index])
        print(predicted_word)
