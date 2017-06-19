# tidy up global environment
rm(list=ls())

#load necessary packages
library(tm)
library(ggplot2)
library(wordcloud)
library(RColorBrewer)
library(NLP)
library(qdap)
library(RTextTools)
library(e1071)
library(RWeka)
library(rJava)
library(stringr)
library(SnowballC)
library(caret)
library(rpart)
library(LiblineaR)
library(stringi)
library(MASS)
library(klaR)
library(kernlab)
library(rminer)
library(topicmodels)
library(lsa)
library(FSelector)

set.seed(123)

#Use this to measure time running on methods
t1 = Sys.time()
print(difftime(Sys.time(), t1, units = 'sec'))
#

###################STEP 1: PREPROCESSING #####################


# getting news_final from Rda
load("Adhoc_news_final.Rda")
# getting sentiment_final from Rda
load("Adhoc_Sentiment_Final.Rda")

# select parameters for descriptive statistics (maybe) and write csv for further use
#adhoc_news <- data.frame(news_final$date, news_final$time, news_final$cutoffmain, news_final$Firsttopic)
#write.csv(adhoc_news, "adhoc_news.csv")

#write cutoff main news to txt file
write.table(news_final$cutoffmain, "cutoffmain.txt", sep="\n", quote=FALSE)
#read.table('cutoffmain.txt', header=FALSE)

# getting sentiment as the direction of abnormal returns
return_direction <- factor(sentiment_final$BinaryReturnDirection)
write.csv(return_direction, "return_direction.csv", quote=FALSE, row.names=F)

# getting sentiment as the direction of abnormal returns
#abnormal_return <- sentiment_final$AbnormalReturn
#write.csv(abnormal_return, "abnormal_return.csv", quote=FALSE)
#return_direction = ifelse(abnormal_return < 0, 0, 1)
prop.table(table(return_direction)) # 45% negative, 54% positive

# Create corpus for adhoc_news using Corpus and VectorSource in tm package
corpus_cutoff <- Corpus(VectorSource(news_final$cutoffmain), readerControl = list(language = "lat"))

# Inspect one piece of news and the respective return direction
writeLines(as.character(corpus_cutoff[[493]])) #quarter report, profit up, revenue double
news_final$Firsttopic[493] # 24
return_direction[493] # negative
sentiment_final$CompanyName[493] # IDS Scheer

writeLines(as.character(corpus_cutoff[[8000]])) #extend subscription period for loan note issue
news_final$Firsttopic[8000] # 4
return_direction[8000] # negative
sentiment_final$CompanyName[8000] # GEOSENTRIC OYJ

###################### CLEAN CORPUS #######################################

# Plain text document
#cleanCorpus <- tm_map(corpus_cutoff, PlainTextDocument)

# Remove emails
removeEmail <- function(x) {str_replace_all(x,"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+", " ")} 
cleanCorpus <- tm_map(corpus_cutoff, content_transformer(removeEmail))
              
# Remove pattern function
removePattern <- content_transformer(function(x, pattern) gsub(pattern," ",x) )
# Remove url
#cleanCorpus <- tm_map(cleanCorpus, removePattern, "(f|ht)tp(s?)://(.*)[.][a-z]+")
cleanCorpus <- tm_map(cleanCorpus, removePattern, "(www.|(f|ht)tp(s?)://)(.*)[.][a-z]+")

# Transform to lower case
cleanCorpus <- tm_map(cleanCorpus, content_transformer(tolower))

# Remove number
cleanCorpus <- tm_map(cleanCorpus, removeNumbers)

# Inspect
writeLines(as.character(cleanCorpus[[6090]]))

# Convert all words "eur" to "euro", "mio" to "million"
replacePattern <- content_transformer(function(x, pattern1, pattern2) gsub(pattern1,pattern2,x) )
cleanCorpus <- tm_map(cleanCorpus, replacePattern, "\\beur\\b", "euro")
cleanCorpus <- tm_map(cleanCorpus, replacePattern, "\\bmio\\b", "million")

# Replace abbreviations with their full text equivalents (e.g. "Sr." becomes "Senior", a.m. to am)
#cleanCorpus <- tm_map(cleanCorpus, content_transformer(replace_abbreviation))
# Replace common symbols with their word equivalents (e.g. "$" becomes "dollar")
cleanCorpus <- tm_map(cleanCorpus, content_transformer(replace_symbol))
# Convert contractions back to their base words (e.g. "shouldn't" becomes "should not")
#cleanCorpus <- tm_map(cleanCorpus, content_transformer(replace_contraction))

# Remove Punctuation
#removePunctuation <- function(x) {str_replace_all(x,"[[:punct:]]", " ")} #regex
#cleanCorpus <- tm_map(cleanCorpus,content_transformer(removePunctuation))
cleanCorpus <- tm_map(cleanCorpus, removePunctuation) #tm

#remove repetitive words/phrases
cleanCorpus <- tm_map(cleanCorpus, removeWords, c("email","fax","tel", "e mail", "phone",
                                                  "end of ad hoc announcement", 
                                                  "end of ad hoc statement", "cdgap", 
                                                  "ad hoc announcement", "ad hoc release", 
                                                  "dgap", "isin", "wkn", "de"))

# Remove words less than 2 characters like s or d after remove '
cleanCorpus <- tm_map(cleanCorpus, removePattern, "*\\b[[:alpha:]]{1,2}\\b*")

# Remove stopword 
cleanCorpus <- tm_map(cleanCorpus, removeWords, stopwords("english"))

# Stem words
cleanCorpus <- tm_map(cleanCorpus, stemDocument, language="english")

# Remove non-letters
#cleanCorpus <- tm_map(cleanCorpus, removePattern, "[^a-zA-Z]")

#Strip Whitespace
cleanCorpus <- tm_map(cleanCorpus, stripWhitespace)

writeLines(as.character(cleanCorpus[[11212]]))

#cleanCorpus <- VCorpus(VectorSource(cleanCorpus))

# save cleanCorpus in dataframe to be able to transfer to Python
cleanCorpus_dataframe <- data.frame(text=unlist(sapply(cleanCorpus, "[", 1)), stringsAsFactors=F)
#colnames(cleanCorpus_dataframe) <- make.names(colnames(cleanCorpus_dataframe))
# write clean Corpus to txt to transfer to Python for doc2vec
write.table(cleanCorpus_dataframe, "adhoc_news_clean.txt", sep="\n", row.names=F,col.names=F, quote=FALSE)


# prepare dictionary LM
require(gdata)
# read in the dictionary and select the columms of positive or negative words
lm_dict = read.xls("LoughranMcDonald_MasterDictionary_2014.xlsx")
lm_neg = lm_dict[,c(1,8)]
lm_pos = lm_dict[,c(1,9)]

#words negative that are not zero, convert to lowercase
lm_neg = lm_neg[lm_neg$Negative != 0,]
lm_neg = lm_neg[,1]
lm_neg <- lapply(lm_neg, tolower)
write.table(lm_neg, "negativity.txt", row.names=F,col.names=F, sep="\n", quote=FALSE)

#words positive that are not zero, convert to lowercase
lm_pos = lm_pos[lm_pos$Positive != 0,]
lm_pos = lm_pos[,1]
lm_pos <- lapply(lm_pos, tolower)
write.table(lm_pos, "positivity.txt", row.names=F,col.names=F, sep="\n", quote=FALSE)

#####################STEP 2: TEXT MINING AND SENTIMENT ANALYSIS#########################


#####################1. Dictionary based Approach using Net Optimism Sentiment###############

# Create Term Document Matrix
tdm <- TermDocumentMatrix(cleanCorpus)
tdm <- as.matrix(tdm)

# Calculate the frequency of words and sort it by frequency
word_freq <- sort(rowSums(tdm),decreasing=TRUE)
word_freq_df <- data.frame(word = names(word_freq),freq=word_freq)
head(word_freq_df, 20)

# Barplot of most frequent terms

p <- ggplot(subset(word_freq_df, freq>9000), aes(x = reorder(word, -freq), y = freq)) +
  geom_bar(stat = "identity") + 
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  labs(x="Words", y = "Frequencies", title = "Most frequent words")

print(p)

# wordcloud
set.seed(1234)

wordcloud(words = word_freq_df$word, freq = word_freq_df$freq, min.freq = 3,
          max.words=300, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))


# building term document matrix using dictionary LM
pos <- as.data.frame(read.csv("positivity.txt", header=FALSE))
tdm.pos <- TermDocumentMatrix(cleanCorpus, list(dictionary=t(pos)))
neg <- as.data.frame(read.csv("negativity.txt", header=FALSE))
tdm.neg <- TermDocumentMatrix(cleanCorpus, list(dictionary=t(neg)))

# Calculate Net Optimism Sentiment metric
# Initialize empty vector to store results
sentiment <- numeric(length(cleanCorpus))
# Iterate over all documents
for (i in 1:length(cleanCorpus)) {
  # Calculate Net-Optimism sentiment
  sentiment[i] <- (sum(tdm.pos[, i]) - sum(tdm.neg[, i]))/sum(tdm[, i])
}
# Inspect sentiment result, which is nx1 matrix where n is # of documents/news 
summary(sentiment)

# Sentiment Classifier using caret package
names(getModelInfo())

# bind sentiment with return_direction into dataframe as input for caret
net_optim_sentiment <- data.frame(y = factor(return_direction), sentiment)

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(net_optim_sentiment$y, p=0.80, list=FALSE)
# select 20% of the data for validation
testing <- net_optim_sentiment[-validation_index,]
# use the remaining 80% of data to training and testing the models
training <- net_optim_sentiment[validation_index,]
nrow(training)
# Run algorithms using 5-fold cross validation
control <- trainControl(method="cv", number=5)
metric <- "Accuracy"

# 
t1 = Sys.time()
# a) linear algorithms
# glm/logistic regression
set.seed(123)
fit.glm <- train(y~., data=training, method="glm", metric=metric, trControl=control)
# lda
#set.seed(123)
#fit.lda <- train(y~., data=training, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART/decision tree
#set.seed(123)
#fit.cart <- train(y~., data=training, method="rpart", metric=metric, trControl=control)
# Naive Bayes
set.seed(123)
fit.nb <- train(y~., data=training, method="nb", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(123)
fit.svm <- train(y~., data=training, method="svmRadial", metric=metric, trControl=control)
#Warnings that probability is 0 for some cases

# summarize accuracy of models
#results <- resamples(list(glm=fit.glm, lda=fit.lda, cart=fit.cart, nb=fit.nb, svm=fit.svm))
#summary(results)

# select the best models: GLM, SVM, NB to perform prediction on testing set

# estimate skill of GLM on the validation dataset
predictions <- predict(fit.glm, testing)
cm <- confusionMatrix(predictions, testing$y)
NOS_glm_accuracy <- cm$overall['Accuracy']
NOS_glm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of SVM on the validation dataset
predictions <- predict(fit.svm, testing)
cm <- confusionMatrix(predictions, testing$y)
NOS_svm_accuracy <- cm$overall['Accuracy']
NOS_svm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of NB on the validation dataset
predictions <- predict(fit.nb, testing)
cm <- confusionMatrix(predictions, testing$y)
NOS_nb_accuracy <- cm$overall['Accuracy']
NOS_nb_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# Combine metrics into one dataframe
NOS_accuracy <- data.frame(NOS_glm_accuracy, NOS_nb_accuracy, NOS_svm_accuracy)
NOS_metrics <- data.frame(NOS_glm_metrics, NOS_svm_metrics, NOS_nb_metrics)
print(difftime(Sys.time(), t1, units = 'sec'))

NOS_accuracy
NOS_metrics

# Remove from global enviroment
rm(training, testing)

# write to csv
write.csv(NOS_metrics, "NOS_metrics.csv")

