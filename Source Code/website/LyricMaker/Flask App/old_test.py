
import winsound
import time
import math

chords = ['a','a#','b','c','c#','d','d#','e','f','f#','g','g#']

frequency_ratio   = lambda i: int(440*(math.pow(2,1/12)**i)) # 2 to power 1 over 12, 440 as this is the tuning standard

chord_name   = lambda i: chords[i%len(chords)] + str(int((i+(9+4*12))/12))

generate_period = lambda tempo: 1/(tempo/60)

generated_notes  = {chord_name(number): frequency_ratio(number) for number in range(-42,60)}

def play_tune(tune, tempo):
    for note in tune.lower().split():
        if note in generated_notes.keys():
            winsound.Beep(generated_notes[note], int(generate_period(tempo)*1000))
        else:
            time.sleep(generate_period(tempo))

# r is a small break between chords
#play( 'c5 c5 g5 c6 c4 c4 g4 c4 r '
#      'c5 c5 g5 c6 c4 c4 g4 c4 r r r ', 180 )

#play( 'g4 c4 g4 d4 g4 c4 d4 g4 r '
 #     'g4 c4 g4 d4 g4 c4 d4 g4 r ', 180 )
import pronouncing
try:
    word=str(input("Enter an adjective plz:"))
except:
    print("oops")
#print(pronouncing.rhymes(word))
new_word=pronouncing.rhymes(word)
#['diming', 'liming', 'priming', 'rhyming', 'timing']

import random

nouns = ("ben", "tom", "alex","tom","ben")
verbs = ("is","is","is","is","is")
adv = ("crazily.", "really", "foolishly.", "merrily.", "occasionally.")
adj = ("noice", "silly", "funny", "odd", "stupid")
num = random.randrange(0,5)
num2=random.randrange(0,5)
#print(nouns[num] + ' ' + verbs[num] + ' ' + adv[num] + ' ' + word)
#print(nouns[num2] + ' ' + verbs[num2] + ' ' + adv[num2] + ' ' + new_word[0])



def generate_lines(no,topic):
    if topic=="Animals":
        first_lyric="mary had a little lamb\n"
    elif topic=="Transport":
        first_lyric="wheels on bus go round and round\n"
    else:
        first_lyric="Jack and Jill went up a hill"
    #second_lyric="little lamb little lamb,"
    #line=first_lyric+second_lyric
    number=int(no)
    number=number-1
    return first_lyric*number

#print(generate_lines(3))



