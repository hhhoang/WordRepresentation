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

############### 2. N-grams: uni-gram, bi-gram, tri-gram with TfIdf ###############

# NGramTokenizer will not work on Corpus, have to force the corpus to VCorpus
# define NGramTokenizer to pass to control in DocumentTermMatrix

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2)) 
UniBigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2)) 
TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3)) 

# read the clean adhoc news and convert to a corpus
cleanCorpus <- read.table("adhoc_news_clean.txt", sep="\n")
cleanCorpus <- VCorpus(VectorSource(cleanCorpus$V1))
# read the return direction into a vector
return_direction <- read.csv("return_direction.csv")
return_direction <- return_direction$x

################# 2.1. Sentiment Classifier Unigram ########################

# Unigram, remove words with less than 2 characters (such as s left over when we removed '), 
# remove sparse terms, join with return_direction and convert to dataframe 
unigram_dtm <- DocumentTermMatrix(cleanCorpus, control = list(wordLengths=c(2, Inf), weighting=weightTfIdf))
inspect(unigram_dtm) #33901 terms
unigram_dtm_sparse <- removeSparseTerms(unigram_dtm, 0.98)
inspect(unigram_dtm_sparse) #914
frequent_terms <- findFreqTerms(unigram_dtm_sparse, 50)

# bind with return direction
unigram_df = as.data.frame(as.matrix(unigram_dtm_sparse))
unigram_df = data.frame(y=factor(return_direction), unigram_df)
colnames(unigram_df) <- make.names(colnames(unigram_df))
unigram_df[1:5, 1:10]

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(unigram_df$y, p=0.80, list=FALSE)
# select 20% of the data for validation
testing <- unigram_df[-validation_index,]
# use the remaining 80% of data to training and testing the models
training <- unigram_df[validation_index,]
nrow(training)
# Run algorithms using 5-fold cross validation
control <- trainControl(method="cv", number=5)
metric <- "Accuracy"


#timing start CV, training and prediction with caret package
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
fit.svm <- train(y~., data=training, method="svmLinear3", metric=metric, trControl=control)
#Warnings that probability is 0 for some cases

# summarize accuracy of models
#results <- resamples(list(glm=fit.glm, lda=fit.lda, cart=fit.cart, nb=fit.nb, svm=fit.svm))
#summary(results)

# estimate skill of GLM on the validation dataset
predictions <- predict(fit.glm, testing)
unigram_glm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of SVM on the validation dataset
predictions <- predict(fit.svm, testing)
unigram_svm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of NB on the validation dataset
predictions <- predict(fit.nb, testing)
unigram_nb_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# unigram_metrics
unigram_metrics <- data.frame(unigram_glm_metrics, unigram_svm_metrics, unigram_nb_metrics)

print(difftime(Sys.time(), t1, units = 'sec'))

# write to csv
write.csv(unigram_metrics, "unigram_metrics.csv")

rm(training, testing)

#################2.2 Sentiment Classifier Bigram #####################

bigram_dtm <- DocumentTermMatrix(cleanCorpus, control = list(tokenize = BigramTokenizer,
                                                             weighting=weightTfIdf,
                                                             wordLengths = c(2, Inf)))
inspect(bigram_dtm) #582492
bigram_dtm_sparse <- removeSparseTerms(bigram_dtm, 0.98)
inspect(bigram_dtm_sparse) # 370
bigram_df <- as.data.frame(as.matrix(bigram_dtm_sparse))
bigram_df <- data.frame(y=factor(return_direction), bigram_df)
colnames(bigram_df) <- make.names(colnames(bigram_df))
bigram_df[1:5, 1:5]

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(bigram_df$y, p=0.80, list=FALSE)
# select 20% of the data for validation
testing <- bigram_df[-validation_index,]
# use the remaining 80% of data to training and testing the models
training <- bigram_df[validation_index,]
nrow(training)
# Run algorithms using 5-fold cross validation
control <- trainControl(method="cv", number=5)
metric <- "Accuracy"

#timing start CV, training and prediction with caret package
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
fit.svm <- train(y~., data=training, method="svmLinear3", metric=metric, trControl=control)
#Warnings that probability is 0 for some cases

# summarize accuracy of models
#results <- resamples(list(glm=fit.glm, lda=fit.lda, cart=fit.cart, nb=fit.nb, svm=fit.svm))
#summary(results)

# estimate skill of GLM on the validation dataset
predictions <- predict(fit.glm, testing)
bigram_glm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of SVM on the validation dataset
predictions <- predict(fit.svm, testing)
bigram_svm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of NB on the validation dataset
predictions <- predict(fit.nb, testing)
bigram_nb_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# bigram metrics 
bigram_metrics <- data.frame(bigram_glm_metrics, bigram_svm_metrics, bigram_nb_metrics)

print(difftime(Sys.time(), t1, units = 'sec'))

bigram_metrics
# write to csv
write.csv(bigram_metrics, "bigram_metrics.csv")

rm(training, testing, predictions)
#####################2.3 Sentiment Classifier Trigram####################

#Trigram
trigram_dtm <- DocumentTermMatrix(cleanCorpus, control = list(tokenize = TrigramTokenizer,
                                                              weighting=weightTfIdf,
                                                              wordLengths = c(2, Inf)))
inspect(trigram_dtm) # 1294815 terms
trigram_dtm_sparse <- removeSparseTerms(trigram_dtm, 0.98)
inspect(trigram_dtm_sparse) # 113 terms
trigram_df<-as.data.frame(as.matrix(trigram_dtm_sparse))
trigram_df<-data.frame(y=factor(return_direction), trigram_df)
colnames(trigram_df) <- make.names(colnames(trigram_df))
trigram_df[1:5, 1:5]

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(trigram_df$y, p=0.80, list=FALSE)
# select 20% of the data for validation
testing <- trigram_df[-validation_index,]
# use the remaining 80% of data to training and testing the models
training <- trigram_df[validation_index,]
nrow(training)
# Run algorithms using 5-fold cross validation
control <- trainControl(method="cv", number=5)
metric <- "Accuracy"

#timing start CV, training and prediction with caret package
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
fit.svm <- train(y~., data=training, method="svmLinear3", metric=metric, trControl=control)
#Warnings that probability is 0 for some cases

# summarize accuracy of models
#results <- resamples(list(glm=fit.glm, lda=fit.lda, cart=fit.cart, nb=fit.nb, svm=fit.svm))
#summary(results)

# estimate skill of GLM on the validation dataset
predictions <- predict(fit.glm, testing)
trigram_glm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of SVM on the validation dataset
predictions <- predict(fit.svm, testing)
trigram_svm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of NB on the validation dataset
predictions <- predict(fit.nb, testing)
trigram_nb_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# trigram metrics
trigram_metrics <- data.frame(trigram_glm_metrics, trigram_svm_metrics, trigram_nb_metrics)

print(difftime(Sys.time(), t1, units = 'sec'))

rm(training, testing, trigram_glm_metrics, trigram_svm_metrics,trigram_nb_metrics, 
   fit.cart, fit.glm, fit.lda, fit.nb, fit.svm, results)
# write to csv
write.csv(trigram_metrics, "trigram_metrics.csv")

trigram_metrics
#######2.4. Sentiment Classifier Uni & Bigram and chi squared feature selection ####################### 

# build DTM with unigram and bigram, tfifd weighting

bigram_tfidf <- DocumentTermMatrix(cleanCorpus, control = list(tokenize = UniBigramTokenizer, 
                                                                   wordLengths = c(2, Inf),
                                                                   weighting=weightTfIdf))
inspect(bigram_tfidf) # 616393 terms
bigram_tfidf_sparse <- removeSparseTerms(bigram_tfidf, 0.98)
inspect(bigram_tfidf_sparse) # 1284

bigram_tfidf <- as.data.frame(as.matrix(bigram_tfidf_sparse))
bigram_tfidf = data.frame(y=factor(return_direction), bigram_tfidf)
colnames(bigram_tfidf) <- make.names(colnames(bigram_tfidf))
bigram_tfidf[1:5, 1:10]

library(FSelector)
# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(bigram_tfidf$y, p=0.80, list=FALSE)
# select 20% of the data for validation
testing <- bigram_tfidf[-validation_index,]
# use the remaining 80% of data to training and testing the models
training <- bigram_tfidf[validation_index,]
nrow(training)

t1 = Sys.time()
#using chi squared weights for the terms for feature selection
chi_squared_weights <- chi.squared(y~., training)
chi_squared_weights
summary(chi_squared_weights) # max 0.06, mean 0.000455
print(difftime(Sys.time(), t1, units = 'sec'))

#decide the cut off number of features, choose 200 terms based on their weights
subset_chi <- cutoff.k(chi_squared_weights, 200)
subset_chi

# training and testing get only feature w.r.t chi squared
training_chi <- data.frame(y=testing, training[subset_chi])
testing = data.frame(y=testing$y, testing[subset_chi])


# Run algorithms using 5-fold cross validation
control <- trainControl(method="cv", number=5)
metric <- "Accuracy"

#timing start CV, training and prediction with caret package
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
fit.svm <- train(y~., data=training, method="svmLinear3", metric=metric, trControl=control)
#Warnings that probability is 0 for some cases

# summarize accuracy of models
#results <- resamples(list(glm=fit.glm, lda=fit.lda, cart=fit.cart, nb=fit.nb, svm=fit.svm))
#summary(results)

print(difftime(Sys.time(), t1, units = 'sec'))
# select the best models: GLM, SVM, NB to perform prediction on testing set

#testing = data.frame(y=testing$y, testing[subset_chi])

t1 = Sys.time()
# estimate skill of GLM on the validation dataset
predictions <- predict(fit.glm, testing)
bigram_tfidf_glm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of SVM on the validation dataset
predictions <- predict(fit.svm, testing)
bigram_tfidf_svm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of NB on the validation dataset
predictions <- predict(fit.nb, testing)
bigram_tfidf_nb_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))
#
bigram_tfidf_metrics <- data.frame(bigram_tfidf_glm_metrics, bigram_tfidf_svm_metrics, bigram_tfidf_nb_metrics)
print(difftime(Sys.time(), t1, units = 'sec'))
bigram_tfidf_metrics
# write to csv
write.csv(bigram_tfidf_metrics, "bigram_tfidf_metrics.csv")

# Save to one dataframe of ngram metrics
ngrams_metrics = data.frame(unigram_metrics, bigram_metrics, trigram_metrics, bigram_tfidf_metrics)
# write to csv
write.csv(ngrams_metrics, "ngrams_metrics.csv")

#
rm(unigram_metrics, bigram_metrics, trigram_metrics, bigram_tfidf_metrics)
