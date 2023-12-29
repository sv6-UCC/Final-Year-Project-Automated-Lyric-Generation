# Python file for generating lyrics using a neural network aswell as using other methods such as LSTM and GRU


import time
import os
import numpy as np
import tensorflow as tensor

file_path = './reduced_pop.txt'
stop_words = ['"',',','(',')','-','[',']','.']

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
    tensor.keras.layers.GRU(gru_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tensor.keras.layers.Dense(vocabulary_size),
  ])
  return model

lyrics_model = createNeuralNet(vocabulary_size = len(sorted_vocabulary), embedding_dimension=embedding_dimension, gru_units=gru_units, batch_size=BATCH_SIZE)

lyrics_model.summary()

def lossOfModel(labels, logits):
  return tensor.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

EPOCHS = 20

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

def lyricsGeneration(lyric_model,index=0,sentence_length=8):
  temperature=0.6
  
  sample_inputs=["hey","when","how","what"]
  firstWord=sample_inputs[index]
  chars_in_first =  [w for w in firstWord.split(' ')]
  check_input = [word2idx[s] for s in chars_in_first]
  
  text_generated = []

  check_input = tensor.expand_dims(check_input, 0)

  lyric_model.reset_states()
  for j in range(sentence_length-1):
      new_lyrics = lyric_model(check_input)

      new_lyrics = tensor.squeeze(new_lyrics, 0)

      new_lyrics = new_lyrics / temperature 
      new_id = tensor.random.categorical(new_lyrics, num_samples=1)[-1,0].numpy()

      check_input = tensor.expand_dims([new_id], 0)
      text_generated.append(' ' + idx2words[new_id])
  

  return (firstWord + ''.join(text_generated))

updated_model.save('final_model.h5') 
a=0
new_list=[]
second_list=[]
third_list=[]
while (a!=4):
  new_list.append(lyricsGeneration(updated_model,index=1))
  a+=1
b=0
while (b!=4):
  second_list.append(lyricsGeneration(updated_model,index=2))
  b+=1
c=0
while (c!=4):
  third_list.append(lyricsGeneration(updated_model,index=3))
  c+=1

print(new_list)
print(second_list)
print(new_list)
print(third_list)
print(new_list)

#temperature of  0.6, produces lyrics not as similar to dataset however this value may change at a later stage
#still need to implement rhyming for sentences
# currently takes around 2 minutes to run due to number of epochs needed for the loss to be less than 1
#However will look into decreasing the time for generation to complete in line with increasing the size of the pop songs textfile
