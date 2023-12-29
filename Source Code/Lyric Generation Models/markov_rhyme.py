import numpy as np
import random
import re
from collections import defaultdict

from time import perf_counter
# Read text from file and tokenize.
count_start_time = perf_counter()
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
  print(first_node)
  print("firstttttt")
  options = list(chain[first_node].keys())

  node_weights = np.array(
      list(chain[first_node].values()),
      dtype=np.float64)

  node_weights /= node_weights.sum()

  print(options)

  print(node_weights)

  new_word = np.random.choice(options, None, p=node_weights)
  
  return [new_word] + move_through_chain(
      chain, first_node=new_word, chain_distance=chain_distance-1,)

def move_through_second_chain(chain, first_node=None, chain_distance=5):
    if not first_node:
        first_node = random.choice(list(chain.keys()))
    
    if chain_distance <= 0:

        return []

    # get options and node weights from the previous two nodes
    prev_nodes = [first_node]
    while len(prev_nodes) < 3:
        
        prev_node = random.choice(list(chain.keys()))
        if prev_node != prev_nodes[-1]:
            prev_nodes.append(prev_node)

    #options = list(chain[tuple(prev_nodes)].keys())
    options=list(chain[prev_node].keys())
    #node_weights = np.array(
     #   list(chain[tuple(prev_nodes)].values()),
      #  dtype=np.float64)
    
    node_weights = np.array(
        list(chain[(prev_node)].values()),
        dtype=np.float64)

    node_weights /= node_weights.sum()



    # choose the next node using the probability weights
    new_word = np.random.choice(options, None, p=node_weights)

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
        
        prev_node = random.choice(list(chain.keys()))
        if prev_node != prev_nodes[-1]:
            prev_nodes.append(prev_node)

    #options = list(chain[tuple(prev_nodes)].keys())
    options=list(chain[prev_node].keys())
    #node_weights = np.array(
     #   list(chain[tuple(prev_nodes)].values()),
      #  dtype=np.float64)
    
    node_weights = np.array(
        list(chain[(prev_node)].values()),
        dtype=np.float64)

    node_weights /= node_weights.sum()



    # choose the next node using the probability weights
    new_word = np.random.choice(options, None, p=node_weights)

    # recursively call the function with the chosen node as the new starting point
    return [new_word] + move_through_third_chain(
        chain, first_node=tuple(prev_nodes[1:] + [new_word]), chain_distance=chain_distance-1)

my_list=[]

from transformers import BertTokenizer, BertForNextSentencePrediction
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from torch.nn.functional import softmax
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

for i in range(1000):
  #first_chain=(' '.join(move_through_chain(
   # markov_graph, chain_distance=100)), '\n')
  #second_chain=(' '.join(move_through_second_chain(
   # markov_graph, chain_distance=100)), '\n')
  third_chain=(' '.join(move_through_third_chain(
    markov_chain, chain_distance=100)), '\n')
  
count_end_time = perf_counter()
count_res=""
print('Count time    : %f %s' % (count_end_time - count_start_time,
                                count_res))