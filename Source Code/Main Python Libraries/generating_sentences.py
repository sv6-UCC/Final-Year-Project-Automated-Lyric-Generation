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


generate()
egg =nlp([nouns[num], verbs[num2]])
generate()
egg2= nlp([verbs[num],adj[num2]])
generate()
egg3= nlp([verbs[num],adj[num2]])
print(egg)
print(egg2)
print(egg3)



#transport

#1 car
car =nlp(["car","powerful"])
car2 =nlp(["wheels","rotate"])
car3 =nlp(["bumpy", "seats"])

print(car)
print(car2)
print(car3)

import enchant
d = enchant.Dict("en_US") #library which includes an English dictionary
lyric_index = random.randint(1, 9)

bad_list=['ahn','abhor','abreu','adl-tabatabai','adair',"ange",'asbill','baz',"birle","birr"]
import pronouncing
from randomsentence.sentence_maker import SentenceMaker
new_sentence="Happy Birthday to me"
last_word1=new_sentence[-1]
pronunciations = pronouncing.rhymes(last_word1)
sentence_maker = SentenceMaker()
sing=""
try:
    if pronunciations[lyric_index] in bad_list:
        lyric_index+=1
        tagged_sentence = sentence_maker.from_keyword_list([pronunciations[lyric_index]])
        print("")
    elif (d.check(pronunciations[0])==False):
        lyric_index+=1
        tagged_sentence = sentence_maker.from_keyword_list([pronunciations[lyric_index]])
    else:
        tagged_sentence = sentence_maker.from_keyword_list([pronunciations[lyric_index]])
        print(tagged_sentence)
    for n in tagged_sentence:
        sing+=n[0]
        sing+=" "
    print(sing)
except Exception as E:
    print(E)
