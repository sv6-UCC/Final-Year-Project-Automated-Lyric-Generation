from lyricsgenius import Genius

token="SfepOmnsT5rtRFUQ3x1yaFivVGdMDh-dJTp27YDJ0WywdZNAfDQby0MWwXACi8SQ"
genius = Genius(token)


# this gets the lyrics of all the songs that have the rock tag.
genius = Genius(token)
page = 1
rock_lyrics = []
i=0
while page:
    res = genius.tag('rock', page=page)
    for hit in res['hits']:
        try:
            song_lyrics = genius.lyrics(song_url=hit['url'])
        except:
            continue
        rock_lyrics.append(song_lyrics)
    page_no=i+1
    print("page " +str(page_no) +" scanned")
    i+=1
    page = res['next_page']
    if i==2:
    #if i==10:
        break

rock_file = open("smallrock.txt", "w", encoding="utf8")
#rock_file = open("rock.txt", "w", encoding="utf8")
for ro_lyric in rock_lyrics:
    rock_file.write(ro_lyric)

"""

# this gets the lyrics of all the songs that have the pop tag.
genius = Genius(token)
page = 1
pop_lyrics = []
i=0
while page:
    print(i)
    res = genius.tag('pop', page=page)
    for hit in res['hits']:
        try:
            song_lyrics = genius.lyrics(song_url=hit['url'])
        except:
            continue
        pop_lyrics.append(song_lyrics)
    i+=1
    page = res['next_page']
    if i==10:
        break

pop_file = open("pop.txt", "w", encoding="utf8")
for p_lyric in pop_lyrics:
    pop_file.write(p_lyric)

"""

"""

# this gets the lyrics of all the songs that have the rap tag.
genius = Genius(token)
page = 1
rap_lyrics = []
i=0
while page:
    print(i)
    res = genius.tag('rap', page=page)
    for hit in res['hits']:
        try:
            song_lyrics = genius.lyrics(song_url=hit['url'])
        except:
            continue
        rap_lyrics.append(song_lyrics)
    i+=1
    page = res['next_page']
    if i==10:
        break

rap_file = open("rap.txt", "w", encoding="utf8")
for r_lyric in rap_lyrics:
    rap_file.write(r_lyric)

"""
