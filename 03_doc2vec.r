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
################## 3. DOC2VEC #####################################

#### Input was exported from Python as csv files to be loaded again in R 
#### for Sentiment Classifier 

######## 3.1 Doc2Vec Distributed Memory ###############################
# read doc2vec_dm 400 300 100
doc2vec_dm <- read.csv("docvecs_dm_400.csv")
# read the return direction into a vector
return_direction <- read.csv("return_direction.csv")
return_direction <- return_direction$x

doc2vec_dm <- data.frame(y=factor(return_direction), doc2vec_dm)
colnames(doc2vec_dm) <- make.names(colnames(doc2vec_dm))
doc2vec_dm[1:5, 1:5]


# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(doc2vec_dm$y, p=0.80, list=FALSE)
# select 20% of the data for validation
testing <- doc2vec_dm[-validation_index,]
# use the remaining 80% of data to training and testing the models
training <- doc2vec_dm[validation_index,]
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
set.seed(123)
#fit.lda <- train(y~., data=training, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART/decision tree
set.seed(123)
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

# estimate skill of GLM on the validation dataset
predictions <- predict(fit.glm, testing)
cm <- confusionMatrix(predictions, testing$y)
doc2vec_dm_glm_accuracy <- cm$overall['Accuracy']
doc2vec_dm_glm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of SVM on the validation dataset
predictions <- predict(fit.svm, testing)
cm <- confusionMatrix(predictions, testing$y)
doc2vec_dm_svm_accuracy <- cm$overall['Accuracy']
doc2vec_dm_svm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of NB on the validation dataset
predictions <- predict(fit.nb, testing)
cm <- confusionMatrix(predictions, testing$y)
doc2vec_dm_nb_accuracy <- cm$overall['Accuracy']
doc2vec_dm_nb_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# Doc2Vec metrics
doc2vec_dm_metrics <- data.frame(doc2vec_dm_glm_metrics, doc2vec_dm_svm_metrics, doc2vec_dm_nb_metrics)
doc2vec_dm_accuracies <- data.frame(doc2vec_dm_glm_accuracy, doc2vec_dm_svm_accuracy, doc2vec_dm_nb_accuracy)

print(difftime(Sys.time(), t1, units = 'sec'))

doc2vec_dm_metrics
doc2vec_dm_accuracies

# write to csv
write.csv(doc2vec_dm_metrics, "doc2vec_dm_400_metrics.csv")

rm(training, testing, predictions, doc2vec_dm_glm_metrics, doc2vec_dm_svm_metrics, doc2vec_dm_nb_metrics,
   fit.glm, fit.nb, fit.svm)

############ 3.2 Doc2Vec Distributed Bag of Words ########################

# proceed the same procedure for doc2vec version distributed bag of word
doc2vec_dbow <- read.csv("docvecs_dbow_400.csv")
dim(doc2vec_dbow) # should be 13135 x 400
doc2vec_dbow <- data.frame(y=factor(return_direction), doc2vec_dbow)
colnames(doc2vec_dbow) <- make.names(colnames(doc2vec_dbow))
doc2vec_dbow[1:5, 1:5]

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(doc2vec_dbow$y, p=0.80, list=FALSE)
# select 20% of the data for validation
testing <- doc2vec_dbow[-validation_index,]
# use the remaining 80% of data to training and testing the models
training <- doc2vec_dbow[validation_index,]
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
fit.svm <- train(y~., data=training, method="svmRadial", metric=metric, trControl=control)
#Warnings that probability is 0 for some cases

# summarize accuracy of models
#results <- resamples(list(glm=fit.glm, lda=fit.lda, cart=fit.cart, nb=fit.nb, svm=fit.svm))
#summary(results)

# estimate skill of GLM on the validation dataset
predictions <- predict(fit.glm, testing)
doc2vec_dbow_glm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of SVM on the validation dataset
predictions <- predict(fit.svm, testing)
doc2vec_dbow_svm_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# estimate skill of NB on the validation dataset
predictions <- predict(fit.nb, testing)
doc2vec_dbow_nb_metrics = mmetric(predictions, testing$y, c("ACC", "TPR", "PRECISION", "F1"))

# 
doc2vec_dbow_metrics <- data.frame(doc2vec_dbow_glm_metrics, doc2vec_dbow_svm_metrics, doc2vec_dbow_nb_metrics)

print(difftime(Sys.time(), t1, units = 'sec'))
write.csv(doc2vec_dbow_metrics, "doc2vec_dbow_400_metrics.csv")

doc2vec_metrics <- data.frame(doc2vec_dm_metrics, doc2vec_dbow_metrics)
# write to csv
write.csv(doc2vec_metrics, "doc2vec_400_metrics.csv")


#
d1 <- read.csv("doc2vec_400_metrics.csv")
d2 <- read.csv("doc2vec_300_metrics.csv")
d3 <- read.csv("doc2vec_100_metrics.csv")
doc2vec_metrics <- data.frame(d1, d2, d3)
write.csv(doc2vec_metrics, "doc2vec_metrics.csv")
