
import nltk

#she wolf by David Guetta
verse1 = nltk.word_tokenize("""A shot in the dark $
A past, lost in space $
And where do I start? $
The past and the chase $
You hunted me down $
Like a wolf, a predator $
I felt like a deer in love lights $
""")

#analysing a random line

verse2 = nltk.word_tokenize("""Thank you for your help $""")
a=nltk.pos_tag(verse1)
print(a)
print("FIRSTTTT")
nn=0
words=[]
new_words=[]
for i in range(0,len(a)):
    if a[i][0] == "$":
        words.append(new_words)
        new_words=[]
        continue
    if a[i][1] =="NN":
        nn+=1
    new_words.append(a[i][1])

b=nltk.pos_tag(verse2)
nn=0
words2=[]
new_words2=[]
for i in range(0,len(b)):
    if b[i][0] == "$":
        words2.append(new_words2)
        new_words2=[]
        continue
    if b[i][1] =="NN":
        nn+=1
    new_words2.append(b[i][1])
#print(nn)
#print(a)
print("Using nltk to discover the types of words in she wolf")
print(words)
print()
print("Doing the same for a different lyric Daniel is traveling tonight on a plane ")
print(words2)

dt_list=['The','This','That','My']
nn_list=["Dog","Cat","Bat","Horse"]
vb_list=["is","was","shows","loves"]
pp_list=['I',"me","mine","myself"]
cc_list=[]

sentence =nltk.word_tokenize("""Thank you for your help""")
sentence_explanation=nltk.pos_tag(sentence)
print()
print(sentence_explanation)

#discovered a Python language which corrects grammer in a sentence

import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
text = 'A sentence with a error'
al=tool.check(text)
print(al)
print("hereee")

#investigating what type of word is in different sentences using Python's nltk module
#for example the word happened is a VBN which stands for a past participle

#https://basicenglishspeaking.com/category/100-sentence-structures/
"""
[('You', 'PRP'), ('only', 'RB'), ('have', 'VBP'), ('to', 'TO'), ('ask', 'VB'), ('her', 'PRP'), ('in', 'IN'), ('order', 'NN'), 
('to', 'TO'), ('know', 'VB'), ('what', 'WP'), ('has', 'VBZ'), ('happened', 'VBN')]

[('You', 'PRP'), ('can', 'MD'), ('never', 'RB'), ('be', 'VB'), ('too', 'RB'), ('careful', 'JJ')]

[('Sure', 'RB'), (',', ','), ('why', 'WRB'), ('not', 'RB'), ('?', '.')]

[('Where', 'WRB'), ('can', 'MD'), ('I', 'PRP'), ('find', 'VB'), ('him', 'PRP'), ('?', '.')]

[('Thank', 'NNP'), ('you', 'PRP'), ('for', 'IN'), ('your', 'PRP$'), ('help', 'NN')]

"""

print()
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
from english_words import english_words_set
ball=list(english_words_set)
j=0
ball2=nltk.pos_tag(ball)
for i in range(0,6000):
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
print()
i=0
song=[]
import pronouncing
while i!=2:
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
    #result+=word
    #try:
        #mylist=pronouncing.rhymes(word)
        #print(mylist[0])
    #except:
        #result=""
        #continue
    song.append(result)
    print(result)
    print()
    fin=tool.correct(result)
    print(fin)
    result=""
    result+=random.choice(nnp_list)
    result+=" "
    result+=random.choice(prp_list)
    result+=" "
    result+=random.choice(in_list)
    result+=" "
    result+=random.choice(prp_list)
    result+=" "
    result+=random.choice(nn_list)
    result+=" "
    #result+=mylist[0]
    song.append(result)
    print(result)
    print()
    fin=tool.correct(result)
    print(fin)
    result=""
    i+=1
print(song)

#some examples I have generated are
#where shall I find him
#why should he maintain them

#i decided to create code to detect iambic pentameter so I can eventually try generate
#lyrics with iambic pentameter

iamb_pat="01"
troc_pat="10"
dac_pat="100"






import syllables

new_sent="Today"
sent_pat = []
for word in new_sent.split():
    pronunciations = pronouncing.phones_for_word(word)
    print(pronunciations)
    try:
        pat = pronouncing.stresses(pronunciations[0])
        print(pat)
        sent_pat.append(pat)
    except:
        continue
print(sent_pat)
result=False
for i in sent_pat:
    if iamb_pat in i:
        result=True
print(result)

print()
print(syllables.estimate(new_sent))
print(pronouncing.syllable_count(new_sent))
