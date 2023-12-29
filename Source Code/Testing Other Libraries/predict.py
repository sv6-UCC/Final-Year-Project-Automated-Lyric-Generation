import torch
from transformers import BertTokenizer, BertForNextSentencePrediction

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model.eval()


def get_logits_and_loss(firstsentence, secondsentence):
    global tokenizer, model

    token_encoding = tokenizer.encode_plus(firstsentence, secondsentence, return_tensors = 'pt')
    token_loss, token_logits = model(**token_encoding, next_sentence_label=torch.LongTensor([1]))

    return token_loss, token_logits

sentence_one = "I was walking to the store one day to buy groceries."
sentence_two = "At the store I bought bananas and milk."

import math

def probability_of_sentences(sentence_one, sentence_two):

    token_loss, token_logits = get_logits_and_loss(sentence_one, sentence_two)
    
    negative_logit = token_logits[0, 1]
    positive_logit = token_logits[0, 0]

    negpart = math.pow(1.2, negative_logit)
    pospart = math.pow(1.2, positive_logit)

    probability = pospart / (pospart + negpart)

    return probability

print(probability_of_sentences(sentence_one,sentence_two))