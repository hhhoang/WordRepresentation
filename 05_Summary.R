
# Putting everything together

# Net Optimism Sentiment
a <- read.csv("NOS_metrics.csv")

# Ngrams
b <- read.csv("ngrams_metrics.csv")

# Doc2Vec
c <- read.csv("doc2vec_metrics.csv")


# getting news_final from Rda
load("Adhoc_news_final.Rda")
# getting sentiment_final from Rda
load("Adhoc_Sentiment_Final.Rda")

datetime <- news_final$dtcreated
heure <- as.integer(substr(datetime,12,13))
conversion <- data.frame(datetime=datetime, heure=heure, period=cut(heure, c(-Inf, 8, 12, 16, Inf),
                                                                    labels=c("before9", "8to13", "13to16","after17")))
