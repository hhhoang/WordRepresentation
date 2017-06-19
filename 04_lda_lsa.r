
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
library(slam)
library(topicmodels)
library(lsa)
library(FSelector)

##################### 4. Topic Models with LDA, LSA  ###############
# The input data for topic models is a document-term matrix
# The tf-idf scores are only used for selecting the vocabulary, 
# the input data consisting of the DTM uses a term-frequency weighting.

# read the clean adhoc news and convert to a corpus
cleanCorpus <- read.table("adhoc_news_clean.txt", sep="\n")
cleanCorpus <- VCorpus(VectorSource(cleanCorpus$V1))
writeLines(as.character(cleanCorpus[[26]]))

unigram <- DocumentTermMatrix(cleanCorpus, control = list(wordLengths = c(2, Inf)))
inspect(unigram) # 25947 terms
unigram_sparse <- removeSparseTerms(unigram, 0.99)
inspect(unigram_sparse) # 1361 terms

findFreqTerms(unigram_sparse, 50)
summary(col_sums(unigram_sparse)) # min 138, mean 1384, max 66685

return_direction <- read.csv("return_direction.csv")
return_direction <- return_direction$x

#LDA doesnt accept rows with only 0
#https://www.jku.at/ifas/content/e108280/e108474/e109445/e142056/Gruen.pdf

term_tfidf <-tapply(unigram_sparse$v/row_sums(unigram_sparse)[unigram_sparse$i],unigram_sparse$j, mean) * 
  log2(nDocs(unigram_sparse)/col_sums(unigram_sparse>0))

summary(term_tfidf) # min 0.01, mean = 0.04, median = 0.04, max = 0.2

# feature selection with tfidf
# include terms which have a tf-idf value of at least 0.03 
# which is a bit less than the median 
# and ensures that the very frequent terms are omitted
unigram_sparse_tfidf <- unigram_sparse[,term_tfidf>=0.03]
unigram_sparse_tfidf <- unigram_sparse_tfidf[row_sums(unigram_sparse_tfidf)>0,]

dim(unigram_sparse_tfidf) #reduce # of terms to 1141
summary(col_sums(unigram_sparse_tfidf))

set.seed(123)

#########choose the  number of topics ###########
# currently doesnt make sense
install.packages("devtools")
devtools::install_github("nikita-moor/ldatuning")
library("ldatuning")

dtm <- unigram_sparse_tfidf[1:1000,]

t1 = Sys.time()
result <- FindTopicsNumber(
  dtm,
  topics = seq(from = 5, to = 40, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)
print(difftime(Sys.time(), t1, units = 'sec'))
FindTopicsNumber_plot(result)

###
t1 = Sys.time()
best.model <- lapply(seq(5,100, by=5), function(k){LDA(dtm, k)})
print(difftime(Sys.time(), t1, units = 'sec'))

best.model.logLik <- as.data.frame(as.matrix(lapply(best.model, logLik)))

best.model.logLik.df <- data.frame(topics=seq(5,100, by=5), LL=as.numeric(as.matrix(best.model.logLik)))
library(ggplot2)
ggplot(best.model.logLik.df, aes(x=topics, y=LL)) + 
  xlab("Number of topics") + ylab("Log likelihood of the model") + 
  geom_line() + 
  theme_bw()  + 
  theme(axis.title.x = element_text(vjust = -0.25, size = 14)) + 
  theme(axis.title.y = element_text(size = 14, angle=90))
#####################

############################ 4.1 LDA ################################
# number of topics
k <- 40 
seed <- 2003 #seed needs to have length of nstart, i.e nstart = 2, seed <-list(1,2)
#Gibbs distribution
# iter, burning, thin control how many Gibbs sampling draws are made
# Gibbs sampling with a burn-in of 1000 iterations 
# and recording every 100th iterations for 1000 iterations.
burnin <- 1000
iter <- 1000 
thin <- 100

nstart <- 1
best <- TRUE # only the best model over all runs

t1 = Sys.time()
unigram_lda <- list(VEM = LDA(unigram_sparse_tfidf, k = k, control = list(seed = seed)), 
                      Gibbs = LDA(unigram_sparse_tfidf, k = k, method = "Gibbs", control = list(nstart=nstart, seed = seed, best=best, burnin=burnin, iter=iter, thin=thin)))
print(difftime(Sys.time(), t1, units = 'sec'))

# access topics, terms, and probabilites of VEM method
topic_vem <- as.matrix(topics(unigram_lda[["VEM"]]))
terms_vem <- terms(unigram_lda[["VEM"]], 10)
#probabilities associated with each topic assignment
probabilites_vem <- as.data.frame(unigram_lda[["VEM"]]@gamma)

# access topics, terms, and probabilites of Gibbs method
topic_gibbs <- topics(unigram_lda[["Gibbs"]])
terms_gibbs <- terms(unigram_lda[["Gibbs"]], 10)
topic_probabilites_gibbs <- as.data.frame(unigram_lda[["Gibbs"]]@gamma)
write.csv(topic_probabilites_gibbs,file=paste("LDAGibbs",k,"TopicProbabilities.csv"))

topic_probabilites_gibbs_dtm <- data.frame(y=factor(return_direction), topic_probabilites_gibbs)

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(topic_probabilites_gibbs_dtm$y, p=0.80, list=FALSE)
# select 20% of the data for validation
testing <- topic_probabilites_gibbs_dtm[-validation_index,]
# use the remaining 80% of data to training and testing the models
training <- topic_probabilites_gibbs_dtm[validation_index,]
nrow(training)
# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
#timing start CV, training and prediction with caret package
t1 = Sys.time()
# a) linear algorithms
# glm/logistic regression
set.seed(123)
fit.glm <- train(y~., data=training, method="glm", metric=metric, trControl=control)
# lda
set.seed(123)
fit.lda <- train(y~., data=training, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART/decision tree
set.seed(123)
fit.cart <- train(y~., data=training, method="rpart", metric=metric, trControl=control)
# Naive Bayes
set.seed(123)
fit.nb <- train(y~., data=training, method="nb", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(123)
fit.svm <- train(y~., data=training, method="svmLinear3", metric=metric, trControl=control)
#Warnings that probability is 0 for some cases

# summarize accuracy of models
results <- resamples(list(glm=fit.glm, lda=fit.lda, cart=fit.cart, nb=fit.nb, svm=fit.svm))
summary(results)

# estimate skill of GLM on the validation dataset
predictions <- predict(fit.glm, testing)
lda_glm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of SVM on the validation dataset
predictions <- predict(fit.svm, testing)
lda_svm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of NB on the validation dataset
predictions <- predict(fit.nb, testing)
lda_nb_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# NOS_accuracies <- data.frame(NOS_glm_accuracy, NOS_svm_accuracy, NOS_nb_accuracy)
lda_metrics <- data.frame(lda_glm_metrics, lda_svm_metrics,lda_nb_metrics)

print(difftime(Sys.time(), t1, units = 'sec'))

# write to csv
write.csv(lda_metrics, "lda_metrics.csv")

###########LSA##########

###########LSA##########
#lsa needs term document matrix

lsa_matrix = t(unigram_sparse_tfidf)
# lsa_matrix = lw_bintf(lsa_matrix) * gw_idf(lsa_matrix) # weighting tfidf
t1 = Sys.time()
LSAspace = lsa(lsa_matrix, dims=dimcalc_share())
print(difftime(Sys.time(), t1, units = 'sec'))

#term vector matrix T (constituting left singular vectors) (orthonormal)
#the document vector matrix D (constituting right singular vectors) (orthonormal), 
#and the diagonal matrix S (constituting singular values).
#M = TSD
#Term vector matrix
lsa_tk=as.data.frame(LSAspace$tk)
# Document vector matrix
lsa_dk=as.data.frame(LSAspace$dk)
#
lsa_sk=as.data.frame(LSAspace$sk)

topic_probabilites_lsa_dtm <- data.frame(y=factor(return_direction), lsa_dk)

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(topic_probabilites_lsa_dtm$y, p=0.70, list=FALSE)
# select 20% of the data for validation
testing <- topic_probabilites_lsa_dtm[-validation_index,]
# use the remaining 80% of data to training and testing the models
training <- topic_probabilites_lsa_dtm[validation_index,]
nrow(training)
# Run algorithms using 5-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#timing start CV, training and prediction with caret package
t1 = Sys.time()
# a) linear algorithms
# glm/logistic regression
set.seed(123)
fit.glm <- train(y~., data=training, method="glm", metric=metric, trControl=control)
# lda
set.seed(123)
fit.lda <- train(y~., data=training, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART/decision tree
set.seed(123)
fit.cart <- train(y~., data=training, method="rpart", metric=metric, trControl=control)
# Naive Bayes
set.seed(123)
fit.nb <- train(y~., data=training, method="nb", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(123)
fit.svm <- train(y~., data=training, method="svmLinear3", metric=metric, trControl=control)
#Warnings that probability is 0 for some cases

# summarize accuracy of models
results <- resamples(list(glm=fit.glm, lda=fit.lda, cart=fit.cart, nb=fit.nb, svm=fit.svm))
summary(results)


# estimate skill of GLM on the validation dataset
predictions <- predict(fit.glm, testing)
lsa_glm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of SVM on the validation dataset
predictions <- predict(fit.svm, testing)
lsa_svm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of NB on the validation dataset
predictions <- predict(fit.nb, testing)
lsa_nb_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

#
lsa_metrics <- data.frame(lsa_glm_metrics, lsa_svm_metrics, lsa_nb_metrics)

print(difftime(Sys.time(), t1, units = 'sec'))

# write to csv
write.csv(lsa_metrics, "lsa_metrics.csv")

topic_modellings_metrics <- data.frame(lda_metrics, lsa_metrics)
# write to csv
write.csv(topic_modellings_metrics, "topic_modellings_metrics.csv")

###############combine all the accuracies of different methods into a table ################

metrics_table <- data.frame(NOS_metrics, 
                             doc2vec_dm_metrics,doc2vec_dbow_metrics,
                             unigram_metrics, bigram_metrics, trigram_metrics, unibigram_tfiff_metrics, 
                             lda_metrics, lsa_metrics)


####### 

#Create document-term matrix
dtm <- unigram_df[,-1]

#convert rownames to filenames
#rownames(dtm) <- filenames

#collapse matrix by summing over columns
freq <- colSums(as.matrix(dtm))
row_sum <- rowSums(as.matrix(dtm))

#length should be total number of terms
length(freq)
#create sort order (descending)
ord <- order(freq,decreasing=TRUE)

#List all terms in decreasing order of freq and write to disk
head(freq[ord])
write.csv(freq[ord],"unigram_freq.csv")

