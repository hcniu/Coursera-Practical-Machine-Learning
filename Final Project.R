library(tidyverse)
library(ggplot2)
library(caret)
library(modelr)
library(randomForest)
library(e1071)
###Read Training Data and clean out the redundant variables
setwd("~/Desktop/R Note/Practical Machine Learning/Final Project")
df<-read.csv("pml-training.csv")
colnames(df)
non<-colnames(df)[str_detect(colnames(df),c("^min_|^var_|^max_|^avg_|^amplitude_|^skewness_|^kurtosis_|^stddev_"))]
df<-select(df,-non)
df<-df[,-c(1,2,3,4,5)]
table(df$classe)

#Clean out highly correlated variables
correlation<-cor(df[,-c(1,55)])
high_cor<-findCorrelation(correlation,cutoff = 0.9)
high_cor<-high_cor+1
df<-df[,-high_cor]

#Subset Validation Dataset
inTrain<-createDataPartition(df$classe,p=0.7,list = F)
testing<-df[-inTrain,]
training<-df[inTrain,]

#Reduce variables: PCA
preProc<-preProcess(training[,-48],method = "pca",thresh = 0.9)
trained<-predict(preProc,training[,-48])
trained<-mutate(trained,"classe"=training$classe)

tested<-predict(preProc,testing[,-48])
tested<-mutate(tested,"classe"=testing$classe)
#First Model: Random Forest
model_rf<-train(classe~.,data = trained,method="rf")
model_final_rf<-model_rf$finalModel
pred_rf<-predict(model_rf,tested)
freq_rf<-table(tested$classe,pred_rf)
confusionMatrix(freq_rf)

#Second Model: Boosting-gbm
model_gbm<-train(classe~.,data = trained,method="gbm",verbose=FALSE)
model_final_gbm<-model_gbm$finalModel
pred_gbm<-predict(model_gbm,tested)
freq_gbm<-table(tested$classe,pred_gbm)
confusionMatrix(freq_gbm)

#Third Model: svm
model_svm<-svm(classe~.,data = trained)
pred_svm<-predict(model_svm,tested,type="class")
freq_svm<-table(tested$classe,pred_svm)
confusionMatrix(freq_svm)

##Final Prediction
#Read Testing Data and Clean out the redundant variables
validation<-read.csv("pml-testing.csv")
non<-colnames(validation)[str_detect(colnames(validation),c("^min_|^var_|^max_|^avg_|^amplitude_|^skewness_|^kurtosis_|^stddev_"))]
validation<-select(validation,-non)
validation<-validation[,-c(1,2,3,4,5)]
validation<-validation[,-high_cor]
validated<-predict(preProc,validation[,-48])
Final_pred<-predict(model_rf,validated)
Final_pred
