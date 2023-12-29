# Python file for generating lyrics using a Markov Chain aswell as using other methods from random_generation.py
# such as language_tool_python and word_tokenize


from nltk.metrics import edit_distance
import re
import numpy as np
from collections import defaultdict
import random
import nltk

file = 'pop.txt'

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
  
  options = list(markov_chain[first_node].keys())

  node_weights = np.array(
      list(markov_chain[first_node].values()),
      dtype=np.float64)

  node_weights /= node_weights.sum()

  new_word = np.random.choice(options, None, p=node_weights)
  
  return [new_word] + move_through_chain(
      chain, first_node=new_word, chain_distance=chain_distance-1,)


no_verse=int(input("Enter the number of versuses you need: "))
verse= int(input("Enter the number of lines in each verse: "))
chorus= int(input("Enter the number of lines in the chorus: "))
line=int(input("Enter the length of each line: "))
chorus_type=str(input("Select the structure of each line in the Chorus - A:[Adverb, Modal, Personal Pronoun, Verb, Personal Pronoun], B:[Personal Pronoun, Adverb, Verb, Adverb, Verb]: "))

part=no_verse*2
final_num=part*verse
total=final_num //2

import pronouncing
from randomsentence.sentence_maker import SentenceMaker

import language_tool_python
tool = language_tool_python.LanguageTool('en-US')


my_list=[]


for i in range(total):
  sentence=(' '.join(move_through_chain(
      markov_chain, chain_distance=line)), '\n')

  lyric=sentence[0]
  print("Length is")
  print(len(lyric.split()))
  new_sentence=""
  other_sentence=""
  first=tool.correct(sentence[0])
  print(sentence[0])
  print()
  print(first)
  new_sentence=nltk.word_tokenize(first)
  last_word1=new_sentence[-1]
  pronunciations = pronouncing.rhymes(last_word1)
  sentence_maker = SentenceMaker()
  try:
    tagged_sentence = sentence_maker.from_keyword_list([pronunciations[0]])
  except:
    total+=1
    continue
  print(tagged_sentence)
  my_list.append(first)
  j=0
  while j==0:
    if (len(tagged_sentence)!= line):
      tagged_sentence = sentence_maker.from_keyword_list([pronunciations[0]])
      continue
    for i in range(0,len(tagged_sentence)):
      #new_sentence+=i[0]
      other_sentence+=tagged_sentence[i][0]
      other_sentence+=" "
    #my_list.append(new_sentence)
    j=1
  my_list.append(other_sentence)
print("Verse:")
print(my_list)


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
    
print()
print(len(nn_list))
print(len(dt_list))
print(len(vb_list))
print(len(pp_list))
print(len(cc_list))
print(len(rb_list))
print(len(in_list))
print(len(jj_list))
print(len(prp_list))
print(len(nns_list))
print(vb_list)
result=""
print("Chorus:")
print()
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

print(song)

