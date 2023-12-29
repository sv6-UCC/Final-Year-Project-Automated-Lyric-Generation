#i decided to create code to detect iambic pentameter so I can eventually try generate
#lyrics with iambic pentameter

iamb_pat="01"
troc_pat="10"
dac_pat="100"

import syllables
import pronouncing
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