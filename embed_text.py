import clean_corpus
import pandas as pd
import gensim.downloader as api
import statistics

model = api.load("glove-wiki-gigaword-300")

# So I have the model in now. I will use this similarity function to find the
# words that are most similar to the words on one side as opposed to the other

# Use the dot product for this??
# Something like:
# ||A|| - ||B||
# -----   -----
# len(A)  len(b)

# Emotion Dictionary
afinn_wl_url = ('https://raw.githubusercontent.com'
                '/fnielsen/afinn/master/afinn/data/AFINN-111.txt')

afinn_wl_df = pd.read_csv(afinn_wl_url,
                          header=None,  # no column names
                          sep='\t',  # tab sepeated
                          names=['term', 'value'])  # new column names

# Negative values are negative words and positive values are positive words.
# There are 2477 words in total. This is gonna be fun to run.

# I just need to clean up the corpus, lowercase all words, remove stop words
# and punctuation. I also need to make sure words already in the dictionary are
# not being included in the final analysis.
corpus = clean_corpus.corpusFiltered

positive = afinn_wl_df[afinn_wl_df["value"] > 0]
negative = afinn_wl_df[afinn_wl_df["value"] < 0]

corpusSimilarity = {'word': [], 'totalValue': [],
                    'positive': [], 'negative': []}
percentage = 0
for w in corpus:
    positiveSim = []
    negativeSim = []

    # Rough indicator of percentage progress
    if corpus.index(w) % 465 == 0:
        percentage += 1
        print("%i percent completed" % percentage)

    for p in positive['term']:
        try:
            positiveSim.append(model.similarity(w, p))
        except KeyError:
            continue

    for n in negative['term']:
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

df = pd.DataFrame(corpusSimilarity)

df.to_csv("data/wordsSimilarity_v1_08292022.csv")

df.sort_values(by=['totalValue'])