import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
def analyse(lyric):
    comp = sid.polarity_scores(lyric)
    comp1 = comp['neg']
    comp2 = comp['neu']
    comp3 = comp['pos']
    if comp1 > 0.5:
        print("this lyric is negative")
    if comp2 > 0.5:
        print("this lyric is neutral")
    if comp3 > 0.5:
        print("this lyric is positive")
    return comp

print(analyse("You can go your own way"))