"""
import random

nouns = ("cat", "dog", "monkey", "horse", "rabbit")
pronouns= ("he", "she", "they")
verbs = ("loves", "makes", "jumps", "feels","smell","hears", "leaves","likes","looks","meets","needs","notices","prefers","provides","reaches","recognizes","appears") 
adv = ("But", "really", "very", "merrily", "occasionally.")
adj = ("fearless","mysterious","beautiful","powerful","bumpy","neverending","windy","appalling","bored","brave","lucky","defiant","horrific","disturbed","livid2")

from keytotext import pipeline
import random
num=1
num2=2
num3=3
def generate():
    global num
    global num2
    global num3
    num =random.randint(0,9)
    num2 =random.randint(0,9)
    num3 =random.randint(0,9)
nlp=pipeline("k2t")

new=nlp(["Russia","leader","Putin"])
print(new)



#animals
# 1 dog
generate()
egg =nlp([nouns[num], verbs[num2]])
generate()
egg2= nlp([verbs[num],adj[num2]])
generate()
egg3= nlp([verbs[num],adj[num2]])
print(egg)
print(egg2)
print(egg3)

#2 cat

cat= nlp(["cat", ",mysterious"])
cat2= nlp(["whiskers","pointy"])
cat3 = nlp(["beautiful","purr"])

print()
print(cat)
print(cat2)
print(cat3)

#transport

#1 car
car =nlp(["car","powerful"])
car2 =nlp(["wheels","rotate"])
car3 =nlp(["bumpy", "seats"])

print(car)
print(car2)
print(car3)

#2 train

train =nlp(["neverending", "train"])
train2 =nlp(["windy","tracks"])
train3 =nlp(["hear","choo choo"])

print()
print(train)
print(train2)
print(train3)

import nltk

sentence1 = "This sentence is a test"
sentence2 = "This sentence is the best"

words1 = nltk.word_tokenize(sentence1)
words2 = nltk.word_tokenize(sentence2)

last_word1 = words1[-1]
last_word2 = words2[-1]

#python -m spacy download en_core_web_sm


from gingerit.gingerit import GingerIt

text = 'so outside oh but i really want to me tight don t know it feels'

parser = GingerIt()
egg=parser.parse(text)
print(egg['result'])

from happytransformer import  HappyTextToText
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
from happytransformer import TTSettings

beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=100)

input_text_1 = "grammar: other side and gold we re beautiful i need to danger s gotta mark my."

output_text_1 = happy_tt.generate_text(input_text_1, args=beam_settings)
print(output_text_1.text)

from paraphraser import paraphrase
# Generate list of sentences with similar contextual meaning
sentences = paraphrase('what is the matter with you?')
print(sentences)

"""
import nltk

text = nltk.word_tokenize("wet rainy day")
print(nltk.pos_tag(text))


# importing the package  
import language_tool_python
from keytotext import pipeline

# using the tool  
my_tool = language_tool_python.LanguageTool('en-US')  
  
# given text  
my_text = """So outside oh, but I really want to me tight don't know it feels"""   
   
# getting the matches  
my_matches = my_tool.check(my_text)  
  
# printing matches  
print(my_matches)

nlp=pipeline("k2t")

new=nlp(["house","big"])
print(new)

from randomsentence.sentence_maker import SentenceMaker

sentence_maker = SentenceMaker()
while(True):
    
    tagged_sentence = sentence_maker.from_keyword_list(["house","big"])
    if(len(tagged_sentence))<9:
        break
print(tagged_sentence)
#despite the sheer house, a spectacle so big
#keep the house big
#this house can be considered big
#https://github.com/patarapolw/randomsentence

from gingerit.gingerit import GingerIt

#text = 'you love I ' wins
text= "you love I"

parser = GingerIt()
egg=parser.parse(text)
print(egg['result'])

print("next")


import language_tool_python

tool = language_tool_python.LanguageTool('en-US')
text = 'This sentence has a error'
first_command=tool.check(text)
print(first_command)

second_command=tool.correct(text)
print(second_command)






#print("you love I" )
print("she knows yet" )

from nltk.corpus import wordnet

synonyms = []

for syn in wordnet.synsets("love"):
    for i in syn.lemmas():
        synonyms.append(i.name())






