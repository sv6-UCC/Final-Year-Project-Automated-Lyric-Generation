# Python file for generating lyrics using a Markov Chain aswell as using other methods from random_generation.py
# such as language_tool_python and word_tokenize


from nltk.metrics import edit_distance
import re
import numpy as np
from collections import defaultdict
import random
import nltk
from transformers import BertTokenizer, BertForNextSentencePrediction
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from torch.nn.functional import softmax
from gingerit.gingerit import GingerIt


global generation
generation=[]
sid = SentimentIntensityAnalyzer()

file="pop.txt"

def choose_dataset(choice):
  global file
  global tokenizer
  global textfile
  global end_word
  global markov_chain
  if choice=="pop":
     file='pop.txt'
  if choice=="rap":
     file='rap.txt'
  if choice=="rock":
     file='rock.txt'
  if choice=="idk":
     file='pop.txt'
  with open(file,encoding="utf8") as tf:
      textfile = tf.read()

  tokenizer = [
      word
      for word in re.split('\W+', textfile)
      if word != ''
  ]

  end_word = tokenizer[0].lower()

  markov_chain = defaultdict(lambda: defaultdict(int))
  for new_word in tokenizer[1:]:
    new_word = new_word.lower()
    markov_chain[end_word][new_word] += 1
    end_word = new_word

with open(file) as tf:
  textfile = tf.read()

tokenizer = [
    word
    for word in re.split('\W+', textfile)
    if word != ''
]
 
end_word = tokenizer[0].lower()

markov_chain = defaultdict(lambda: defaultdict(int))
for new_word in tokenizer[1:]:
  new_word = new_word.lower()
  markov_chain[end_word][new_word] += 1
  end_word = new_word


def move_through_chain(chain, first_node=None,chain_distance=5):


  if not first_node:
    first_node = random.choice(list(chain.keys()))

  if chain_distance <= 0:
    return []
  print("first word in markov chain:")
  print(first_node)
  options = list(markov_chain[first_node].keys())

  node_weights = np.array(
      list(markov_chain[first_node].values()),
      dtype=np.float64)

  node_weights /= node_weights.sum()
  print("next possible word choices")
  print(options)

  new_word = np.random.choice(options, None, p=node_weights)
  print(new_word + " is chosen")
  return [new_word] + move_through_chain(
      chain, first_node=new_word, chain_distance=chain_distance-1,)

def move_through_second_chain(chain, first_node=None, chain_distance=5):
    if not first_node:
        first_node = random.choice(list(chain.keys()))

    if chain_distance <= 0:
        return []

    print("first word in markov chain:")
    print(first_node)

    # get options and node weights from the previous two nodes
    prev_nodes = [first_node]
    while len(prev_nodes) < 3:
        print(prev_nodes)
        prev_node = random.choice(list(chain.keys()))
        if prev_node != prev_nodes[-1]:
            prev_nodes.append(prev_node)

    print("the words previously were")
    print(prev_nodes)
    #options = list(chain[tuple(prev_nodes)].keys())
    options=list(chain[prev_node].keys())
    #node_weights = np.array(
     #   list(chain[tuple(prev_nodes)].values()),
      #  dtype=np.float64)
    
    node_weights = np.array(
        list(chain[(prev_node)].values()),
        dtype=np.float64)

    node_weights /= node_weights.sum()

    print("next possible word choices")
    print(options)

    # choose the next node using the probability weights
    new_word = np.random.choice(options, None, p=node_weights)
    print(new_word + " is chosen")

    # recursively call the function with the chosen node as the new starting point
    return [new_word] + move_through_second_chain(
        chain, first_node=tuple(prev_nodes[1:] + [new_word]), chain_distance=chain_distance-1)

def move_through_third_chain(chain, first_node=None, chain_distance=5):
    if not first_node:
        first_node = random.choice(list(chain.keys()))

    if chain_distance <= 0:
        return []
    
      # get options and node weights from the previous two nodes
    prev_nodes = [first_node]
    while len(prev_nodes) < 4:
        print(prev_nodes)
        prev_node = random.choice(list(chain.keys()))
        if prev_node != prev_nodes[-1]:
            prev_nodes.append(prev_node)

    print(prev_nodes)
    #options = list(chain[tuple(prev_nodes)].keys())
    options=list(chain[prev_node].keys())
    #node_weights = np.array(
     #   list(chain[tuple(prev_nodes)].values()),
      #  dtype=np.float64)
    
    node_weights = np.array(
        list(chain[(prev_node)].values()),
        dtype=np.float64)

    node_weights /= node_weights.sum()

    print(options)


    print(node_weights)


    # choose the next node using the probability weights
    new_word = np.random.choice(options, None, p=node_weights)

    # recursively call the function with the chosen node as the new starting point
    return [new_word] + move_through_second_chain(
        chain, first_node=tuple(prev_nodes[1:] + [new_word]), chain_distance=chain_distance-1)


#no_verse=int(input("Enter the number of versuses you need: "))
#verse= int(input("Enter the number of lines in each verse: "))
#chorus= int(input("Enter the number of lines in the chorus: "))
#line=int(input("Enter the length of each line: "))
#chorus_type=str(input("Select the structure of each line in the Chorus - A:[Adverb, Modal, Personal Pronoun, Verb, Personal Pronoun], B:[Personal Pronoun, Adverb, Verb, Adverb, Verb]: "))

#part=no_verse*2
#final_num=part*verse

#total=final_num //2

import pronouncing
from randomsentence.sentence_maker import SentenceMaker

import language_tool_python
tool = language_tool_python.LanguageTool('en-US')




bad_list=['ahn','acquit','abou','abhor','avant','a.d','abreu','adl-tabatabai',"birle","i"]

import enchant
d = enchant.Dict("en_US")
tokenizer_new = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
lyric_index = random.randint(3, 12)


def output_list():
   global generation
   return generation

def createfirstorder(how4):
  return move_through_chain(
          markov_chain, chain_distance=int(how4))

def createsecondorder(how4):
   return move_through_second_chain(
    markov_chain, chain_distance=int(how4))

def createthirdorder(how4):
   return move_through_third_chain(
    markov_chain, chain_distance=int(how4))

parser = GingerIt()
def generate_verse(how,how3,how4,lyric_type,mark1,eff,how5,mark):
  my_song=[]
  global generation
  how=int(how)
  how3=int(how3)
  part=how3*2
  if lyric_type == "chorus":
     how=how*2
  elif lyric_type =="verse":
     how=how
  final_num=part*how
  total=final_num //2
  sing=""
  for i in range(how):
    my_dict={}
    if lyric_type=="verse":
      if mark =="first":
          chain=createfirstorder(how4)
      if mark =="second":
          chain=createsecondorder(how4)
      if mark =="third":
          chain=createthirdorder(how4)
      else:
          chain=createsecondorder(how4)
      sentence=(' '.join(chain), '\n')
      lyric=sentence[0]
      
      generation.append("Length is")
      real_length=len(lyric.split())
      new_sentence=""
      other_sentence=""
      other_first=parser.parse(sentence[0])
      first=other_first['result']
      #first=tool.correct(sentence[0])
      true=first
      print(first +" is chosen")
      
      c=0
      while c!=4:
        if how5=="no" or how5=="idk":
          if mark =="first":
            chain=createfirstorder(how4)
          if mark =="second":
            chain=createsecondorder(how4)
          else:
            chain=createsecondorder(how4)
          sentence=(' '.join(chain), '\n')
          lyric=sentence[0]
        if how5=="ye":
          new_sentence=nltk.word_tokenize(first)
          last_word1=new_sentence[-1]
          pronunciations = pronouncing.rhymes(last_word1)
          sentence_maker = SentenceMaker()
          try:
            tagged_sentence = sentence_maker.from_keyword_list([pronunciations[lyric_index]])
          except:
            sentence=(' '.join(chain), '\n')
            lyric=sentence[0]
            
            generation.append("Length is")
            real_length=len(lyric.split())
            new_sentence=""
            other_sentence=""
            other_first=parser.parse(sentence[0])
            first=other_first['result']
            #first=tool.correct(sentence[0])
            true=first
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
              how+=1
              continue
        if mark1 =="bad":
           if comp1 <= comp3:
              print("positive score")
              print(comp3)
              print("negative score")
              print(comp1)
              how+=1
              continue

        embeddings = tokenizer_new.encode_plus(first, lyric, return_tensors='pt')
        logits = model(**embeddings)[0]
        probs = softmax(logits, dim=1)
        print("probability of "+lyric+" following "+ first)
        print(probs[0][0].item())
        my_dict[lyric] =probs[0][0].item()
        c+=1
      max_value=0
      selected_line=""
      print(my_dict)
      for key,value in my_dict.items():
        if value>max_value:
          max_value=value
          selected_line=key
      
      generation.append("Length is")
      #first=tool.correct(selected_line)
      print(selected_line + " is chosen")
      other_first=parser.parse(selected_line)
      first=other_first['result']
      my_song.append(true)
      my_song.append(first)
      continue
    print(sentence[0])
    print()
    print(first)
    new_sentence=nltk.word_tokenize(first)
    last_word1=new_sentence[-1]
    pronunciations = pronouncing.rhymes(last_word1)
    sentence_maker = SentenceMaker()
    try:
      if pronunciations[0] in bad_list:
        tagged_sentence = sentence_maker.from_keyword_list([pronunciations[1]])
        print("")
      elif (d.check(pronunciations[0])==False):
        tagged_sentence = sentence_maker.from_keyword_list([pronunciations[1]])
      else:
        tagged_sentence = sentence_maker.from_keyword_list([pronunciations[0]])
    except:
      total+=1
      continue
    print(tagged_sentence)
    my_song.append(first)
    j=0
    while j==0:
      if (len(tagged_sentence)!= 6):
        tagged_sentence = sentence_maker.from_keyword_list([pronunciations[0]])
        continue
      for i in range(0,len(tagged_sentence)):
        #new_sentence+=i[0]
        other_sentence+=tagged_sentence[i][0]
        other_sentence+=" "
      #my_list.append(new_sentence)
      j=1
    my_song.append(other_sentence)
  print("Verse:")
  print(my_song)
  
 

  # return string
  return my_song


rb_list=[]
in_list=[]
jj_list=[]
prp_list=[]
nns_list=[]
vbp_list=[]
wrb_list=[]
vbd_list=[]
vbz_list=[]
vbg_list=[]
md_list=[]
nnp_list=[]
dt_list=[]
nn_list=[]
vb_list=[]
pp_list=[]
cc_list=[]

#python -m pip install pip==21.3.1
from english_words import english_words_set
ball=list(english_words_set)
j=0
ball2=nltk.pos_tag(ball)
for i in range(0,4000):
  val = ball2[i][1]
  if val =="NN":
      nn_list.append(ball2[i][0])
  if val =="DT":
      dt_list.append(ball2[i][0])
  if val =="VB":
      vb_list.append(ball2[i][0])
  if val =="PP":
      pp_list.append(ball2[i][0])
  if val =="CC":
      cc_list.append(ball2[i][0])
  if val =="RB":
      rb_list.append(ball2[i][0])
  if val =="IN":
      in_list.append(ball2[i][0])
  if val =="JJ":
      jj_list.append(ball2[i][0])
  if val =="PRP":
      prp_list.append(ball2[i][0])
  if val =="NNS":
      nns_list.append(ball2[i][0])
  if val =="VBP":
      vbp_list.append(ball2[i][0])
  if val =="VBD":
      vbd_list.append(ball2[i][0])
  if val =="VBZ":
      vbz_list.append(ball2[i][0])
  if val =="VBG":
      vbg_list.append(ball2[i][0])
  if val =="WRB":
      wrb_list.append(ball2[i][0])
  if val =="NNP":
      nnp_list.append(ball2[i][0])
  if val =="MD":
      md_list.append(ball2[i][0])

import random
    

result=""

def generate_chorus():
  chorus=4
  chorus_type="A"
  for x in range(chorus):
    i=0
    song=[]
    import pronouncing
    if chorus_type == "A":
      result+=random.choice(wrb_list)
      result+=" "
      result+=random.choice(md_list)
      result+=" "
      result+=random.choice(prp_list)
      result+=" "
      result+=random.choice(vb_list)
      result+=" "
      result+=random.choice(prp_list)
      result+=""

    if chorus_type == "B":
      result+=random.choice(prp_list)
      result+=" "
      result+=random.choice(rb_list)
      result+=" "
      result+=random.choice(vbp_list)
      result+=" "
      result+=random.choice(rb_list)
      result+=" "
      result+=random.choice(vb_list)
      result+=" "
    #result+=word
    #try:
        #mylist=pronouncing.rhymes(word)
        #print(mylist[0])
    #except:
        #result=""
        #continue
    song.append(result)
    fin=tool.correct(result)
    result=""



