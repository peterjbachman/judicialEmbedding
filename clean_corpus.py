import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle

df = pd.read_csv("data/opinionText.csv")

testText = df["text"]

corpus = ""
for i in range(0, 368):
    text = testText[i]
    corpus += str(text) + " "

# Tokenize the corpus
# nltk.download('punkt')
# THIS TOOK LIKE 30 MIN TO RUN. BE SURE WHEN YOU ARE RERUNNING IT
corpusTokens = word_tokenize(corpus)

# gonna save raw corpus just in case I mess something up
# with open("data/corpusRaw.bin", "wb") as output:
#     pickle.dump(corpusTokens, output)
# Raw corpus has 3,545,421 words. Need to clean this up.

# Lowercase
corpusFiltered = [word.lower() for word in corpusTokens]

# Remove punctuation
corpusFiltered = [word for word in corpusFiltered if word.isalpha()]

# Remove stop words
# nltk.download('stopwords')
stops = nltk.corpus.stopwords.words('english')
stops.append('§')
corpusFiltered = [word for word in corpusFiltered if not word in stops]

len(corpusFiltered)  # 1,398,758 words in corpus

# Save full cleaned corpus
# with open("data/corpusFiltered_08292022.bin", "wb") as output:
#     pickle.dump(corpusFiltered, output)

# Keep only unique words in corpus
corpusSet = set(corpusFiltered)
corpusFiltered = list(corpusSet)

len(corpusFiltered)  # 33,679 unique words

# save corpus of unique words
with open("data/corpusFilteredUnique_08292022.bin", "wb") as output:
    pickle.dump(corpusFiltered, output)
