library(wordcloud2)
library(tidyverse)
library(htmlwidgets)
library(webshot)

textCorpus <- read.csv("data/wordsSimilarityAfinn.csv") %>%
    select(c("word", "count", "totalValue")) %>%
    arrange(desc(totalValue))

colors <- ifelse(textCorpus$totalValue > 0, "red",
    ifelse(textCorpus$totalValue < 0, "blue", "black")
)
len <- length(textCorpus$word)
posNeg <- c(1:100, (len - 100):len)

wordPlot <- wordcloud2(textCorpus[posNeg, 1:2],
    color = colors[posNeg], rotateRatio = 0,
    size =1
)

wordPlot
saveWidget(wordPlot, "tmp.html", selfcontained = FALSE)

# and in png or pdf
webshot(
    url = "tmp.html",
    file = "plots/wordCloud.pdf",
    delay = 240,
    vwidth = 4000,
    vheight = 2000,
    selector = '#canvas'
)
