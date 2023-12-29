# Python file for generating lyrics using a neural network aswell as using other methods such as LSTM and GRU


import time
import os
import numpy as np
import tensorflow as tensor
from random import randint
import nltk

file_path = './newfile.txt'
stop_words = ['"',',','(',')','-','[',']','.',":"]

def choose_dataset(choice):
  global file
  if choice=="pop":
     filepath='pop.txt'
  if choice=="rap":
     filepath='rap.txt'
  if choice=="rock":
     filepath='rap.txt'
  if choice=="idk":
     filepath='newfile.txt'

def fileToList(text_file):
  converted_list = [word for word in text_file.split(' ')] 
  converted_list = [clutter for clutter in converted_list if clutter] 
  return converted_list

def textPreprocessing(text_file):
  text_file = text_file.replace('\t','').replace('\n', ' ').replace('\r', ' ')
  processed_file = text_file.lower()
  for character in stop_words:
    processed_file = processed_file.replace(character,' ')
  return processed_file

text_file = open(file_path, 'rb').read().decode(encoding='utf-8')
text_file = textPreprocessing(text_file)
word_list = fileToList(text_file) 

map(str.strip, word_list)

sorted_vocabulary = sorted(set(word_list))

def createInputOutputText(bundle):
  input_text = bundle[:-1]
  output_text = bundle[1:]
  return input_text, output_text

print(sorted_vocabulary)
print("bottleeee")
idx2words = np.array(sorted_vocabulary)
word2idx = {word: index for index, word in enumerate(sorted_vocabulary)}
max_length = 10

indexed_word = np.array([word2idx[original_word] for original_word in word_list])

BUFFER_SIZE = 100 
BATCH_SIZE = 64 

dataset_of_words = tensor.data.Dataset.from_tensor_slices(indexed_word)
dataset_length=max_length+1

embedding_dimension = 256

gru_units = 1024

vocabulary_size = len(sorted_vocabulary)

sequencesOfWords = dataset_of_words.batch(dataset_length, drop_remainder=True) 

sequence_dataset = sequencesOfWords.map(createInputOutputText)

sequence_dataset = sequence_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def createNeuralNet(vocabulary_size, embedding_dimension, gru_units, batch_size):
  model = tensor.keras.Sequential([
    tensor.keras.layers.Embedding(vocabulary_size, embedding_dimension,
                              batch_input_shape=[batch_size, None]),
    tensor.keras.layers.SimpleRNN(embedding_dimension, activation="tanh", return_sequences=True), #simpleRNN the fastest
    tensor.keras.layers.Dense(vocabulary_size),
    #Commented out this line for now but will try implement later
    #tensor.keras.layers.LSTM(4, input_shape=(1, 1))
  ])
  return model



"""

def createNeuralNet(vocabulary_size, embedding_dimension, gru_units, batch_size):
  model = tensor.keras.Sequential([
    tensor.keras.layers.Embedding(vocabulary_size, embedding_dimension,
                              batch_input_shape=[batch_size, None]),
    tensor.keras.layers.LSTM(embedding_dimension, activation="tanh", return_sequences=True),
    tensor.keras.layers.Dense(vocabulary_size),
    #Commented out this line for now but will try implement later
    #tensor.keras.layers.LSTM(4, input_shape=(1, 1))
  ])
  return model

  def createNeuralNet(vocabulary_size, embedding_dimension, gru_units, batch_size):
  model = tensor.keras.Sequential([
    tensor.keras.layers.Embedding(vocabulary_size, embedding_dimension,
                              batch_input_shape=[batch_size, None]),
    tensor.keras.layers.GRU(gru_units,
                        return_sequences=True,
                        stateful=True,
                        activation="tanh", #tanh is used for RNN
                        recurrent_initializer='glorot_uniform'),
    tensor.keras.layers.Dense(vocabulary_size),
    #Commented out this line for now but will try implement later
    #tensor.keras.layers.LSTM(4, input_shape=(1, 1))
  ])
  return model



def createNeuralNet(vocabulary_size, embedding_dimension, gru_units, batch_size):
  model = tensor.keras.Sequential([
    tensor.keras.layers.Embedding(vocabulary_size, embedding_dimension,
                              batch_input_shape=[batch_size, None]),
    tensor.keras.layers.SimpleRNN(embedding_dimension, activation="tanh", return_sequences=True), #simpleRNN the fastest
    tensor.keras.layers.Dense(vocabulary_size),
    #Commented out this line for now but will try implement later
    #tensor.keras.layers.LSTM(4, input_shape=(1, 1))
  ])
  return model

"""



lyrics_model = createNeuralNet(vocabulary_size = len(sorted_vocabulary), embedding_dimension=embedding_dimension, gru_units=gru_units, batch_size=BATCH_SIZE)

lyrics_model.summary()

def lossOfModel(labels, logits):
  return tensor.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

EPOCHS = 40

check_dir = './checkpoints'
lyrics_model.compile(optimizer='adam', loss=lossOfModel)

checkpoint_prefix = os.path.join(check_dir, "check_{epoch}")

call_checkpoint=tensor.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

lyrics_history = lyrics_model.fit(sequence_dataset, epochs=EPOCHS, callbacks=[call_checkpoint])



tensor.train.latest_checkpoint(check_dir)

updated_model = createNeuralNet(len(sorted_vocabulary), embedding_dimension, gru_units, batch_size=1)

updated_model.load_weights(tensor.train.latest_checkpoint(check_dir))

updated_model.build(tensor.TensorShape([1, None]))
updated_model.summary()

with open("starter.txt", "r") as f:
    huge_list = f.read().split()

def lyricsGeneration(lyric_model,index=0,sentence_length=5):
  temperature=0.6
  while True:
    index=randint(0,84)
    sample_inputs=huge_list
    try:
      firstWord=sample_inputs[index]
      chars_in_first =  [w for w in firstWord.split(' ')]
      check_input = [word2idx[s] for s in chars_in_first]
      break
    except:
      continue
  
  text_generated = []

  check_input = tensor.expand_dims(check_input, 0)

  lyric_model.reset_states()
  for j in range(int(sentence_length)-1):
      new_lyrics = lyric_model(check_input)

      new_lyrics = tensor.squeeze(new_lyrics, 0)

      new_lyrics = new_lyrics / temperature 
      new_id = tensor.random.categorical(new_lyrics, num_samples=1)[-1,0].numpy()

      check_input = tensor.expand_dims([new_id], 0)
      text_generated.append(' ' + idx2words[new_id])
  

  return (firstWord + ''.join(text_generated))
updated_model.save('final_model.h5') 
new_list=[]
third_list=[]
from transformers import BertTokenizer, BertForNextSentencePrediction
from torch.nn.functional import softmax
tokenizer_new = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
import pronouncing
from randomsentence.sentence_maker import SentenceMaker

def analyse(lyric,mark1):
    comp = sid.polarity_scores(lyric)
    print(lyric)
    comp1 = comp['neg']
    comp2 = comp['neu']
    comp3 = comp['pos']
    if mark1 =="good":
      if comp3 <= comp1:
          print("positive score")
          print(comp3)
          print("negative score")
          print(comp1)
          return "rip"
    if mark1 =="bad":
      if comp1 <= comp3:
          print("positive score")
          print(comp3)
          print("negative score")
          print(comp1)
          return "rip"
def generate_RNNverse(third2,first2,mark1,how5):
  a=0
  new_list=[]
  third_list=[]
  times=int(third2)
  no_times=times//2
  for a in range(no_times):
    while True:
      lyric2=lyricsGeneration(updated_model,index=0,sentence_length=first2)
      result=analyse(lyric2,mark1)
      if result=="rip":
        continue
      break
    
    true=lyric2
    my_dict={}
    c=0
    while c!=4:
      if how5=="ye":
        new_sentence=nltk.word_tokenize(lyric2)
        last_word1=new_sentence[-1]
        pronunciations = pronouncing.rhymes(last_word1)
        sentence_maker = SentenceMaker()
        try:
          tagged_sentence = sentence_maker.from_keyword_list([pronunciations[0]])
        except:
          while True:
            lyric2=lyricsGeneration(updated_model,index=0,sentence_length=first2)
            result=analyse(lyric2,mark1)
            if result=="rip":
              continue
            break
          true=lyric2
          continue
        if (len(tagged_sentence)< 4 or len(tagged_sentence) >9):
          continue
        other_sentence=""
        try:
          for i in range(0,len(tagged_sentence)):
            print(tagged_sentence)
            #new_sentence+=i[0]
            other_sentence+=tagged_sentence[i][0]
            other_sentence+=" "
        except:
          continue
        lyric=other_sentence
      else:
        lyric=lyricsGeneration(updated_model,index=0,sentence_length=first2)
      embeddings = tokenizer_new.encode_plus(lyric2, lyric, return_tensors='pt')
      logits = model(**embeddings)[0]
      probs = softmax(logits, dim=1)
      print("probability of "+lyric+" following "+ lyric2)
      print(probs[0][0].item())
      my_dict[lyric] =probs[0][0].item()
      c+=1
    max_value=0
    selected_line=""
    for key,value in my_dict.items():
      if value>max_value:
        max_value=value
        selected_line=key
    print(selected_line + " is chosen")
    new_list.append(true)
    new_list.append(selected_line)
    a+=1
  return new_list
def getModel():
  return updated_model

  

#temperature of  0.6, produces lyrics not as similar to dataset however this value may change at a later stage
#still need to implement rhyming for sentences
# currently takes around 2 minutes to run due to number of epochs needed for the loss to be less than 1
#However will look into decreasing the time for generation to complete in line with increasing the size of the pop songs textfile