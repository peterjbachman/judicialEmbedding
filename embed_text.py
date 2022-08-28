import nltk
import pandas as pd
import gensim.downloader as api

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
# THere are 2477 words in total. This is gonna be fun to run.

model.similarity()
