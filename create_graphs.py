import pandas as pd
import matplotlib.pyplot as plt

afinnSim = pd.read_csv("data/wordsSimilarityAfinn.csv")
bingSim = pd.read_csv("data/wordsSimilarityBing.csv")

afinnSimFiltered = afinnSim[afinnSim["totalValue"] != 0]
bingSimFiltered = bingSim[bingSim["totalValue"] != 0]

plt.style.use('bmh')
fig, ax = plt.subplots(nrows=2, sharex=True)

# Show Afinn Plot
ax[0].scatter(x=afinnSimFiltered['totalValue'], y=afinnSimFiltered['count'])
ax[0].axvline(0, linestyle='--', color="gray")
ax[0].set_title("Afinn Sentiment Dictionary")
ax[0].set_ylabel('Word Count')

# Show Bing Plot
ax[1].scatter(x=bingSimFiltered['totalValue'], y=bingSimFiltered['count'])
ax[1].axvline(0, linestyle='--', color="gray")
ax[1].set_title("Bing Sentiment Dictionary")
ax[1].set_xlabel("Average Similarity to Positive and Negative Sentiment")
ax[1].set_ylabel('Word Count')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.5)
plt.savefig("plots/sentimentPlot.pdf")

# Create Histograms
fig, ax = plt.subplots(nrows=2, sharex=True)
n_bins = 20

# Show Afinn Plot
ax[0].hist(afinnSimFiltered['totalValue'], bins=n_bins)
ax[0].axvline(0, linestyle='--', color="gray")
ax[0].set_title("Afinn Sentiment Dictionary")

# Show Bing Plot
ax[1].hist(bingSimFiltered['totalValue'], bins=n_bins)
ax[1].axvline(0, linestyle='--', color="gray")
ax[1].set_title("Bing Sentiment Dictionary")
ax[1].set_xlabel("Average Similarity to Positive and Negative Sentiment")

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.5)
plt.savefig("plots/sentimentHist.pdf")
