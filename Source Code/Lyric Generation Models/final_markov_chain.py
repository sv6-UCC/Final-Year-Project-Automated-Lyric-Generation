# Python file for generating lyrics using a Markov Chain aswell as using other methods from random_generation.py
# such as language_tool_python and word_tokenize



import re
import numpy as np
from collections import defaultdict
import random
import nltk
import pronouncing
from randomsentence.sentence_maker import SentenceMaker
from transformers import BertTokenizer, BertForNextSentencePrediction
from torch.nn.functional import softmax
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import enchant
from english_words import english_words_set
#import language_tool_python
from gingerit.gingerit import GingerIt

file = 'pop.txt'

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

import numpy as np
import random

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



no_verse=int(input("Enter the number of versuses you need as an integer: "))
verse= int(input("Enter the number of lines in each verse as an integer: "))
chorus= int(input("Enter the number of lines in the chorus as an integer: "))
line=int(input("Enter the length of each line as an integer: "))
scheme=int(input("Choose a rhyming scheme:1:AABB,2:ABAB,3:ABBA"))
chain_choice=int(input("1:First-Order, 2:Second-Order, 3:Third-Order"))
rhyme=str(input("Do you want end rhyme?: Type yes or type no"))
lyric_sentiment=int(input("What type of lyrics: 1: Positive, 2:Negative"))
#chorus_type=str(input("Select the structure of each line in the Chorus - A:[Adverb, Modal, Personal Pronoun, Verb, Personal Pronoun], B:[Personal Pronoun, Adverb, Verb, Adverb, Verb]: "))

part=no_verse*2
final_num=part*verse
total=final_num //2

lyric_index = random.randint(3, 12)


#tool = language_tool_python.LanguageTool('en-US')
parser = GingerIt()

my_list=[]

bad_list=['a','ab','ahn','acquit','abou','abhor',"affine",'avant','a.d','abreu','adl-tabatabai','adair','ai','asbill','baz',"birle","ange","birr","bing"]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
sid = SentimentIntensityAnalyzer()


d = enchant.Dict("en_US")
zero="y"
one="y"
two="y"
three="y"
my_counter=0
rhyme_list=[1,3,5,7,9,11,13]
if scheme ==3:
  total=total*2
for i in range(total):
  while True:
    a=0
    if chain_choice ==1:
        sentence=(' '.join(move_through_chain(
        markov_chain, chain_distance=line)), '\n')
    if chain_choice ==2:
        sentence=(' '.join(move_through_second_chain(
        markov_chain, chain_distance=line)), '\n')
        print(sentence)
    if chain_choice ==3:
        sentence=(' '.join(move_through_third_chain(
        markov_chain, chain_distance=line)), '\n')
        print(sentence)
    if scheme ==3:
        
        print(my_list)
        if i in rhyme_list:
            if my_counter==0:
                zero=first
                three=tagged_sentence
                
                my_counter=1
                continue
            elif my_counter==1:
                one=first
                two=tagged_sentence
                my_list.append(zero)
                my_list.append(one)
                my_list.append(two)
                my_list.append(three)
                zero="y"
                one="y"
                two="y"
                three="y"
                my_counter=0
                continue
    lyric=sentence[0]

    other_first=parser.parse(sentence[0])

    first=other_first['result']
    comp = sid.polarity_scores(first)
    print(first)
    print(comp)
    comp1 = comp['neg']
    comp2 = comp['neu']
    comp3 = comp['pos']
    if comp1>0.5 and lyric_sentiment==2:
        print("This lyric has negative sentiment")
        break
    elif comp2>0.5:
        print("This lyric has neutral sentiment")
    elif comp3>0.5 and lyric_sentiment==1:
        print("This lyric has positive sentiment")
        break


  my_dict={}
  while a!=3:
    if rhyme=="yes":
      new_sentence=nltk.word_tokenize(first)
      last_word1=new_sentence[-1]
      pronunciations = pronouncing.rhymes(last_word1)
      sentence_maker = SentenceMaker()
      sing=""
      try:
          if pronunciations[lyric_index] in bad_list:
              lyric_index-=1
              tagged_sentence = sentence_maker.from_keyword_list([pronunciations[lyric_index]])
              
          elif (d.check(pronunciations[0])==False):
            lyric_index-=1
            tagged_sentence = sentence_maker.from_keyword_list([pronunciations[lyric_index]])
          else:
            tagged_sentence = sentence_maker.from_keyword_list([pronunciations[lyric_index]])
            print(tagged_sentence)
          for n in tagged_sentence:
            sing+=n[0]
            sing+=" "
          print(sing)
          embeddings = tokenizer.encode_plus(first, sing, return_tensors='pt')
          logits = model(**embeddings)[0]
          probs = softmax(logits, dim=1)
          
          print(probs[0][0].item())
          my_dict[sing] =probs[0][0].item()
      except Exception as E:
        print(E)
        total+=1

      a+=1
      continue
    else:
        if chain_choice ==1:
            sentence=(' '.join(move_through_chain(
            markov_chain, chain_distance=line)), '\n')
        if chain_choice ==2:
            sentence=(' '.join(move_through_second_chain(
            markov_chain, chain_distance=line)), '\n')
            
            print(sentence)
        other_first=parser.parse(sentence[0])
        #print(other_first['result'])
        sing=other_first['result']
        #my_list.append(first)
        embeddings = tokenizer.encode_plus(first, sing, return_tensors='pt')
        logits = model(**embeddings)[0]
        probs = softmax(logits, dim=1)
        
        print(probs[0][0].item())
        my_dict[sing] =probs[0][0].item()
        a+=1
        
        continue
  max_value=0
  selected_line=""
  for key,value in my_dict.items():
     if value>max_value:
        max_value=value
        selected_line=key
  tagged_sentence=selected_line
  print(my_dict)
  lyric=sentence[0]
  lyric_index=0
  new_sentence=""
  other_sentence=""
  if scheme==1:
    my_list.append(first)
    j=0
    my_list.append(tagged_sentence)
  
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

ball=list(english_words_set)
j=0
ball2=nltk.pos_tag(ball)
"""
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
  #song.append(result)
  fin=tool.correct(result)
  result=""

print(song)
"""
