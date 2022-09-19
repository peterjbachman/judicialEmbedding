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
ax[0].axvline(0, linestyle='--', color="black")
ax[0].set_title("Afinn")
ax[0].set_ylabel('Word Count')

# Show Bing Plot
ax[1].scatter(x=bingSimFiltered['totalValue'], y=bingSimFiltered['count'])
ax[1].axvline(0, linestyle='--', color="black")
ax[1].set_title("Bing")
ax[1].set_xlabel("Average Similarity to Positive and Negative Sentiment")
ax[1].set_ylabel('Word Count')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.5)
plt. savefig("plots/sentimentPlot.pdf")

# Create Histograms
fig, ax = plt.subplots()
n_bins = 20

# Show Afinn Plot
ax.hist(afinnSimFiltered['totalValue'], bins=n_bins, alpha=0.5, label="Afinn")
ax.axvline(0, linestyle='--', color="black")
ax.set_xlabel("Average Similarity to Positive and Negative Sentiment")

# Show Bing Plot
ax.hist(bingSimFiltered['totalValue'], bins=n_bins, alpha=0.5, label="Bing")
plt.legend(loc="upper right")
ratio = 0.55
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
plt.tight_layout()
plt.savefig("plots/sentimentHist.pdf")

bingSimFiltered.sort_values(by=["totalValue"])
afinnSimFiltered.sort_values(by=["totalValue"])

# filter for words that show up more than 50 times
afinnSimFiltered = afinnSim[afinnSim["count"] <= 20]
bingSimFiltered = bingSim[bingSim["count"] <= 20]

# New scatter plot
fig, ax = plt.subplots()

# Show Afinn Plot
ax.scatter(x=bingSimFiltered['totalValue'], y=bingSimFiltered['count'])
ax.axvline(0, linestyle='--', color="black")
ax.set_ylabel('Word Count')
ax.set_xlabel("Average Sentiment")

ratio = 0.55
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
plt.tight_layout()
plt.savefig("plots/sentimentAfinnRare.pdf")
