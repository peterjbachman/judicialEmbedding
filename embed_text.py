from collections import Counter
import pandas as pd
import gensim.downloader as api
import statistics
import pickle

model = api.load("glove-wiki-gigaword-300")

# So I have the model in now. I will use this similarity function to find the
# words that are most similar to the words on one side as opposed to the other

# Use the dot product for this??
# Something like:
# ||A|| - ||B||
# -----   -----
# len(A)  len(b)

# Emotion Dictionary
# I may try and find another one or two to see if there are differences
# I could also try weighting the similarity based on the value of the sentiment
afinn_wl_url = ('https://raw.githubusercontent.com'
                '/fnielsen/afinn/master/afinn/data/AFINN-111.txt')

afinn_wl_df = pd.read_csv(afinn_wl_url,
                          header=None,  # no column names
                          sep='\t',  # tab sepeated
                          names=['term', 'value'])  # new column names

# Negative values are negative words and positive values are positive words.
# There are 2477 words in total. This is gonna be fun to run.

# Bing Sentiment Dictionary
bingPos = open("data/positive-words.txt", "r+")
bingPos = bingPos.read().split("\n")

bingNeg = open("data/negative-words.txt", "r+")
bingNeg = bingNeg.read().split("\n")

# Oh also I need to remove any words that are in either dictionary

corpus = open("data/corpusFiltered_08292022.bin", "rb")
corpus = pickle.load(corpus, encoding="bytes")
counts = Counter(corpus)
corpus = list(set(corpus))

positive = afinn_wl_df[afinn_wl_df["value"] > 0]
negative = afinn_wl_df[afinn_wl_df["value"] < 0]


def calcSimilarity(corpus, positive, negative, wordCounts):
    corpusSimilarity = {'word': [], 'totalValue': [],
                        'positive': [], 'negative': [], 'count': []}
    percentage = 0
    fraction = len(corpus) // 100
    for w in corpus:
        positiveSim = []
        negativeSim = []
        # Rough indicator of percentage progress
        if corpus.index(w) % fraction == 0:
            percentage += 1
            print("%i percent completed" % percentage)

        # Skip calculating similarity if word is in sentiment dictionary
        if (w in positive) or (w in negative):
            print("Word %s is in the sentiment dictionary. Skipping." % w)
            continue

        # Calculate similarity to positive words, if either word is not found,
        # then skip that word
        for p in positive:
            try:
                positiveSim.append(model.similarity(w, p))
            except KeyError:
                continue

        # Calculate similarity to negative words, if either word is not found,
        # then skip that word
        for n in negative:
            try:
                negativeSim.append(model.similarity(w, n))
            except KeyError:
                continue

        # Append dictionary
        if len(positiveSim) < 1:
            meanPositive = 0
        else:
            meanPositive = statistics.mean(positiveSim)

        if len(negativeSim) < 1:
            meanNegative = 0
        else:
            meanNegative = statistics.mean(negativeSim)
        diff = meanPositive - meanNegative

        corpusSimilarity["word"].append(w)
        corpusSimilarity["totalValue"].append(diff)
        corpusSimilarity["positive"].append(meanPositive)
        corpusSimilarity["negative"].append(meanNegative)
        corpusSimilarity["count"].append(wordCounts[str(w)])

    df = pd.DataFrame(corpusSimilarity)

    return df


# Use Afinn sentiment dictionary
afinnSim = calcSimilarity(corpus, list(
    positive['term']), list(negative['term']), counts)
afinnSim.to_csv("data/wordsSimilarityAfinn_v3_08302022.csv")
afinnSim.sort_values(by=['totalValue'])

# Use Bing Sentiment Dictionary
bingSim = calcSimilarity(corpus, bingPos, bingNeg, counts)
bingSim.to_csv("data/wordsSimilarityBing_v1_08302022.csv")
bingSim.sort_values(by=['negative'])
