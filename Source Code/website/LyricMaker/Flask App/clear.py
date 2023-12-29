infile = "allgenres.txt"
outfile = "newfile.txt"

delete_list = ["word_1","chorus","CHORUS","Chorus","[","]","Verse","Intro","()","(",")","Outro","Interlude","Pre-","Instrumental","187EmbedTranslationsEspañolEnglish"]
with open(infile,encoding="utf8") as fin, open(outfile, 'w+', encoding='utf-8') as fout:
    for line in fin:
        for word in delete_list:
            line = line.replace(word, "")
        fout.write(line)