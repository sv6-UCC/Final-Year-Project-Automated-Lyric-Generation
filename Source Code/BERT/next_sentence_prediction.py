original_lyric="Is this the real life?"
options=["Just killed a man","I don't want to die","Is this just fantasy?","Nothing really matters to me"]
my_dict={}
from transformers import BertTokenizer, BertForNextSentencePrediction
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from torch.nn.functional import softmax
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
j=0
while j<4:
    embeddings = tokenizer.encode_plus(original_lyric, options[j], return_tensors='pt')
    logits = model(**embeddings)[0]
    probs = softmax(logits, dim=1)
    my_dict[options[j]] =probs[0][0].item()
    j+=1
max_value=0
selected_line=""
for key,value in my_dict.items():
    if value>max_value:
        max_value=value
        selected_line=key
tagged_sentence=selected_line
print(my_dict)
print("1st lyric is "+original_lyric)
print("predicted 2nd lyric is " +tagged_sentence)