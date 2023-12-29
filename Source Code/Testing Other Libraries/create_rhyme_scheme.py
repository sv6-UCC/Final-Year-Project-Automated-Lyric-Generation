import nltk
import enchant
import pronouncing
from randomsentence.sentence_maker import SentenceMaker
import random
scheme="AABB"
#scheme="ABAB"
#scheme="ABBA"

z=0
d = enchant.Dict("en_US")
lyric_index = random.randint(1, 9)
rhyme_list=[]
bad_list=['a','ab','ahn','acquit','abou','abhor',"affine",'avant','a.d','abreu','adl-tabatabai','adair','ai','asbill','baz',"birle","ange","birr","bing"]
while z<2:
    if z==0:
        first_sentence="The heart is a bloom"
        rhyme_list.append(first_sentence)
    if z==1:
        first_sentence="See the stone set in your eyes"
        rhyme_list.append(first_sentence)
    new_sentence=nltk.word_tokenize(first_sentence)
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
        rhyme_list.append(sing)
        z+=1
    except Exception as E:
        print(E)
print(scheme)
if scheme =="AABB":
    print(rhyme_list[0])
    print(rhyme_list[1])
    print(rhyme_list[2])
    print(rhyme_list[3])
if scheme =="ABAB":
    print(rhyme_list[0])
    print(rhyme_list[2])
    print(rhyme_list[1])
    print(rhyme_list[3])
if scheme =="ABBA":
    print(rhyme_list[0])
    print(rhyme_list[2])
    print(rhyme_list[3])
    print(rhyme_list[1])



