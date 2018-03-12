#Load data
SET.SEED(200)
train=read.csv(file="C:/Users/Deepti/Dropbox/Deepti - Blood Donation Modeling/train.csv", header=TRUE, sep=",")
str(train)
names(train)
# Subset the attitude data
dat = train[,c(2,3,5)]

# Check for the optimal number of clusters given the data
# Determine number of clusters
mydata=dat
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(mydata, 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

# K-Means Cluster Analysis
fit <- kmeans(mydata, 5) # 6 cluster solution
# get cluster means 
aggregate(mydata,by=list(fit$cluster),FUN=mean)
# append cluster assignment
mydata <- data.frame(mydata, fit$cluster)

# Perform K-Means with the optimal number of clusters identified from the Elbow method
set.seed(7)
km2 = kmeans(dat, 5, nstart=100)

# Examine the result of the clustering algorithm
km2

# Plot results
plot(dat, col =(km2$cluster +1) , main="K-Means result with 5 clusters", pch=20, cex=2)

#Fitting models on different clusters
#Print the clustering vector
km2$cluster
table(km2$cluster)

#combine the results of cluster with main data
train1=cbind(train,km2$cluster)
str(train1)
names(train1)
library(plyr)
train1 <- rename(train1,c('km2$cluster'='Cluster'))
#summary of data by clasters
table(train1$Cluster)
#Run model on cluster1
clust1=subset(train1,train1$Cluster==1)
names(clust1)
#Divide the dataset into training and testing and run each model
library(caret)
library(plyr)
library(ipred)
library(e1071)
library(pROC)
library(gbm)
library(survival)
library(splines)
library(parallel)
library(randomForest)
library(MASS)
library(e1071)
summary(clust1)
clust1$X=NULL

clust1$Cluster=NULL
names(clust1)
rch.2007)
#find the roc curve on testing data
rocCurvec5 <- roc (response = testdf$Made.Donation.in.March.2007, predictor = c5Probs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvec5, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("C5"), fill = c("red"))
auc(rocCurvec5)
#CART
library(caret)
library(rpart)
cartmod1 <- train(Made.Donation.in.March.2007~.,
                  traindf,
                  method = "rpart",
                  preProc = c("center","scale"),  # Center and scale data
                  metric="ROC",
                  trControl=ctrl)
summary(cartmod1)

#Predict on training data
cartProbs1 <- predict(cartmod1, newdata=traindf, type="prob")[,1]
cartClasses1 <- predict(cartmod1, newdata=traindf)
confusionMatrix(cartClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on traininging data
library(pROC)

rocCurvecart1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = cartProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecart1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Cart"), fill = c("red"))
auc(rocCurvecart1)# split data set into training and testinG
set.seed(2016)
split <- createDataPartition(clust1$Made.Donation.in.March.2007, p=0.7, list=F)
traindf <- clust1[split,]
testdf <-  clust1[-split,]
names(traindf)
#Model with all attributes
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
glmfit1=train(Made.Donation.in.March.2007~.,data=traindf,  method = "glm", family = "binomial", trControl = ctrl, metric = "ROC")
summary(glmfit1)

#Model without Total Volume Donated
# Example of Logisic Regression algorithms
names(traindf)
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
logit <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "glm", family = "binomial", trControl = ctrl, metric = "ROC")
summary(logit)
#Finding confusion Matrix for Trainingg Data

logitProbs1 <- predict(logit, newdata=traindf, type="prob")[,1]
logitClasses1 <- predict(logit, newdata=traindf)
confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve for TRAININGng Data

rocCurve1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = logitProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurve1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Logit"), fill = c("red"))
auc(rocCurve1)

#Finding confusion Matrix for Testing Data

logitProbs <- predict(logit, newdata=testdf, type="prob")[,1]
logitClasses <- predict(logit, newdata=testdf)
confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve for Testing Data

rocCurve <- roc (response = testdf$Made.Donation.in.March.2007, predictor = logitProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurve, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Logit"), fill = c("red"))
auc(rocCurve)


#Bagging

bagmod=train(Made.Donation.in.March.2007~.,data=traindf,method="treebag",trControl=trainControl(method="cv",number=5))
#inding consusion matrix on training data
bagProbs1 <- predict(bagmod, newdata=traindf, type="prob")[,1]
bagClasses1 <- predict(bagmod, newdata=traindf)
confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on training data
rocCurvebag1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = bagProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvebag1, le1gacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Bagging"), fill = c("red"))
auc(rocCurvebag1)

#inding consusion matrix on testing data
bagProbs <- predict(bagmod, newdata=testdf, type="prob")[,1]
bagClasses <- predict(bagmod, newdata=testdf)
confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvebag <- roc (response = testdf$Made.Donation.in.March.2007, predictor = bagProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvebag, le1gacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Bagging"), fill = c("red"))
auc(rocCurvebag)

#Boosting

boostmod=train(Made.Donation.in.March.2007~.,data=traindf,method="gbm",verbose=F,trControl=trainControl(method="cv",number=60))
#Creating confusion matrix for training data
boostProbs1 <- predict(boostmod, newdata=traindf, type="prob")[,1]
boostClasses1 <- predict(boostmod, newdata=traindf)
confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve for training data
rocCurveboost1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = boostProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurveboost1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Boosting"), fill = c("red"))
auc(rocCurveboost1)
#Creating confusion matrix for testing data
boostProbs <- predict(boostmod, newdata=testdf, type="prob")[,1]
boostClasses <- predict(boostmod, newdata=testdf)
confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve for testing data
rocCurveboost <- roc (response = testdf$Made.Donation.in.March.2007, predictor = boostProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurveboost, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Boosting"), fill = c("red"))
auc(rocCurveboost)
#RandomForest 

RFmod=train(Made.Donation.in.March.2007~.,data=traindf,method="rf",importance=T,trControl=trainControl(method="cv",number=50))
#finding confusion matrix for training data
rfProbs1 <- predict(RFmod, newdata=traindf, type="prob")[,1]
rfClasses1 <- predict(RFmod, newdata=traindf)
confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on training data
rocCurverf1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = rfProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurverf1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Random Forest"), fill = c("red"))
auc(rocCurverf1)
#finding confusion matrix for testing data
rfProbs <- predict(RFmod, newdata=testdf, type="prob")[,1]
rfClasses <- predict(RFmod, newdata=testdf)
confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurverf <- roc (response = testdf$Made.Donation.in.March.2007, predictor = rfProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurverf, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Random Forest"), fill = c("red"))
auc(rocCurverf)
#n fold cross validation
# define training control

# 10-fold CV
ctrl <- trainControl(method = "cv", number=10,classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
CVmod <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "nb",tuneLength =10,family = "binomial", trControl = ctrl, metric = "ROC")

# summarize results
summary(CVmod)
#Predict on training data
cvProbs1 <- predict(CVmod, newdata=traindf, type="prob")[,1]
cvClasses1 <- predict(CVmod, newdata=traindf)
confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve ib training data

rocCurvecv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = cvProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("10 Fold Cross validation"), fill = c("red"))
auc(rocCurvecv1)
#Predict on testing data
cvProbs <- predict(CVmod, newdata=testdf, type="prob")[,1]
cvClasses <- predict(CVmod, newdata=testdf)
confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve ib testing data

rocCurvecv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = cvProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("10 Fold Cross validation"), fill = c("red"))
auc(rocCurvecv)
#Logitboost
install.packages("caTools")
library(caTools)
ctrl <- trainControl(classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
logitboostmod <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "LogitBoost",family = "binomial", tuneLength=50,trControl = ctrl, metric = "ROC")

# summarize results
summary(logitboostmod)
#Predict on training data
lbProbs1 <- predict(logitboostmod, newdata=traindf, type="prob")[,1]
lbClasses1 <- predict(logitboostmod, newdata=traindf)
confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelb1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = lbProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelb1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LogitBoost"), fill = c("red"))
auc(rocCurvelb1)
#Predict on testing data
lbProbs <- predict(logitboostmod, newdata=testdf, type="prob")[,1]
lbClasses <- predict(logitboostmod, newdata=testdf)
confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelb <- roc (response = testdf$Made.Donation.in.March.2007, predictor = lbProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelb, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LogitBoost"), fill = c("red"))
auc(rocCurvelb)

#LDA
ctrl <- trainControl(classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
ldamod <- train(Made.Donation.in.March.2007~.,data = traindf, method = "lda",family = "binomial",trControl = ctrl, metric = "ROC")
ldamod
traindf$Total.Volume.Donated..c.c..=NULL
# summarize results
summary(ldamod)
#Predict on training data
ldaProbs1 <- predict(ldamod, newdata=traindf, type="prob")[,1]
ldaClasses1 <- predict(ldamod, newdata=traindf)
confusionMatrix(data=ldaClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelda1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = ldaProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelda1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LDA"), fill = c("red"))
auc(rocCurvelda1)
#Predict on testing data
ldaProbs <- predict(ldamod, newdata=testdf, type="prob")[,1]
ldaClasses <- predict(ldamod, newdata=testdf)
confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvelda <- roc (response = testdf$Made.Donation.in.March.2007, predictor = ldaProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelda, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LDA"), fill = c("red"))
auc(rocCurvelda)

#svm RBF
# Training SVM Models
library(caret)
library(dplyr)         # Used by caret
library(kernlab)       # support vector machine 
library(pROC)	       # plot the ROC curves
trainX=traindf[,1:3]
names(trainX)
# Setup for cross validation
ctrl <- trainControl(method="repeatedcv",   # 10fold cross validation
                     repeats=5,		    # do 5 repititions of cv
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE)
svcvmod <- train(x=trainX,
                 y= traindf$Made.Donation.in.March.2007,
                 method = "svmRadial",   # Radial kernel
                 tuneLength = 9,					# 9 values of the cost function
                 preProc = c("center","scale"),  # Center and scale data
                 metric="ROC",
                 trControl=ctrl)
svcvmod
#Predict on training data
svcvProbs1 <- predict(svcvmod, newdata=trainX, type="prob")[,1]
svcvClasses1 <- predict(svcvmod, newdata=trainX)
confusionMatrix(data=svcvClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvesvcv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = svcvProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesvcv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesvcv1)
#Predict on testing data
names(testdf)
testdf$Total.Volume.Donated..c.c..=NULL
testX=testdf[,1:3]
names(testX)
svcvProbs <- predict(svcvmod, newdata=testX, type="prob")[,1]
svcvClasses <- predict(svcvmod, newdata=testX)
confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvesvcv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = svcvProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesvcv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesvcv)
#SVM wihout crossvalidation
ctrl <- trainControl(summaryFunction=twoClassSummary,	classProbs=TRUE)
svmod1 <- train(x=trainX,
                y= traindf$Made.Donation.in.March.2007,
                method = "svmRadial",   # Radial kernel
                tuneLength = 9,					# 9 values of the cost function
                preProc = c("center","scale"),  # Center and scale data
                metric="ROC",
                trControl=ctrl)
svmod1

#Predict on training data
svProbs1 <- predict(svmod1, newdata=trainX, type="prob")[,1]
svClasses1 <- predict(svmod1, newdata=trainX)
confusionMatrix(data=svClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvesv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = svProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesv1)
#Predict on testing data
svProbs <- predict(svmod1, newdata=testX, type="prob")[,1]
svClasses <- predict(svmod1, newdata=testX)
confusionMatrix(data=svClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvesv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = svProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesv)

#

#ANN MLP
#install.packages("RSNNS")
library(RSNNS)
ctrl <- trainC1
control(summaryFunction=twoClassSummary,	classProbs=TRUE)
mlpmod1 <- train(Made.Donation.in.March.2007~.,
                 traindf,
                 method = "mlp",
                 preProc = c("center","scale"),  # Center and scale data
                 metric="ROC",
                 trControl=ctrl)
mlpmod1

#Predict on training data
mlpProbs1 <- predict(mlpmod1, newdata=trainX, type="prob")[,1]
mlpClasses1 <- predict(mlpmod1, newdata=trainX)
confusionMatrix(mlpClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvemlp1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = mlpProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvemlp1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("MLP"), fill = c("red"))
auc(rocCurvemlp1)
#Predict on testing data
mlpProbs <- predict(mlpmod1, newdata=testX, type="prob")[,1]
mlpClasses <- predict(mlpmod1, newdata=testX)
confusionMatrix(mlpClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvemlp <- roc (response = testdf$Made.Donation.in.March.2007, predictor = mlpProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvemlp, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("MLP"), fill = c("red"))
auc(rocCurvemlp)

#PNN


#c5.0
install.packages("C50")
library(C50)
library(caret)
traindf
names(traindf)
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
#c5mod1 = train(Made.Donation.in.March.2007~.,trainmethod = "C5.0", preProc = c("center","scale"),metric="ROC")
c5mod1 <- train(Made.Donation.in.March.2007~.,
                traindf,
                method = "C5.0",
                preProc = c("center","scale"),  # Center and scale data
                metric="ROC",
                trControl=ctrl)

#Predict on training data
C5Probs1 <- predict(c5mod1, newdata=traindf, type="prob")[,1]
c5Classes1 <- predict(c5mod1, newdata=traindf)
confusionMatrix(c5Classes1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvec51 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = C5Probs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvec51, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("C5"), fill = c("red"))
auc(rocCurvec51)
#Predict on testing data
textX=testdf[,1:3]
textX
c5Probs <- predict(c5mod1, newdata=testdf, type="prob")[,1]
c5Classes <- predict(c5mod1, newdata=testdf)
confusionMatrix(c5Classes, testdf$Made.Donation.in.Ma
#Predict on testing data
cartProbs <- predict(cartmod1, newdata=testdf, type="prob")[,1]
cartClasses <- predict(cartmod1, newdata=testdf)
confusionMatrix(cartClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvecart <- roc (response = testdf$Made.Donation.in.March.2007, predictor = cartProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecart, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("CART"), fill = c("red"))
auc(rocCurvecart)

# plot ROC curves for training 
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve1, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Training Data")
lines(rocCurvebag1, col="blue")
lines(rocCurveboost1, col="orange")
lines(rocCurverf1, col="light green")
lines(rocCurvecv1,col="yellow")
lines(rocCurvelb1,col="dark green")
lines(rocCurvelda1,col="light blue")
lines(rocCurvesvcv1,col="brown")
lines(rocCurvesv1,col="black")
lines(rocCurvemlp1,col="dark grey")
lines(rocCurvecart1,col="pink")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.5
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost", "LDA","SVM (RBF)"
                  ,"SVM","ANN MLP", "CART")
       , fill=c("red","blue","orange","light green","yellow","dark green","light blue","brown","black"
                ,"dark grey","pink"))
# plot ROC curves for testing
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Testing Data")
lines(rocCurvebag, col="blue")
lines(rocCurveboost, col="orange")
lines(rocCurverf, col="light green")
lines(rocCurvecv,col="yellow")
lines(rocCurvelb,col="dark green")
lines(rocCurvelda,col="light blue")
lines(rocCurvesvcv,col="brown")
lines(rocCurvesv,col="black")
lines(rocCurvemlp,col="grey")
lines(rocCurvecart,col="pink")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.5
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost", "LDA","SVM (RBF)"
                  ,"SVM","ANN MLP", "CART")
       , fill=c("red","blue","orange","light green","yellow","dark green","light blue","brown","black","grey","pink"))


#Storing Results in same Matrix fir Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve)[1])
m2 = cbind(confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag)[1])
m3 = cbind(confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost)[1])
m4 = cbind(confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf)[1])
m5 = cbind(confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv)[1])
m6 = cbind(confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m7 = cbind(confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m8 = cbind(confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m9 = cbind(confusionMatrix(data=svClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m10 = cbind(confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
m11 = cbind(confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
results <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results) <- models
names(results) <- c(stats)
results


#Storing Results in same Matrix for Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve1)[1])
m2 = cbind(confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag1)[1])
m3 = cbind(confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost1)[1])
m4 = cbind(confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf1)[1])
m5 = cbind(confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv1)[1])
m6 = cbind(confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb1)[1])
m7 = cbind(confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m8 = cbind(confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m9 = cbind(confusionMatrix(data=svClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m10 = cbind(confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
m11 = cbind(confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
results1 <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results1) <- models
names(results1) <- c(stats)
results1

#Run model on cluster2
clust2=subset(train1,train1$Cluster==2)
names(clust2)
#Divide the dataset into training and testing and run each model
library(caret)
library(plyr)
library(ipred)
library(e1071)
library(pROC)
library(gbm)
library(survival)
library(splines)
library(parallel)
library(randomForest)
library(MASS)
library(e1071)
summary(clust2)
clust2$X=NULL
clust2$Total.Volume.Donated..c.c..= NULL
clust2$Cluster=NULL
#DROP TRAINDF
traindf=NULL
# split data set into training and testinG
set.seed(2016)
split <- createDataPartition(clust2$Made.Donation.in.March.2007, p=0.7, list=F)
traindf <- clust2[split,]
testdf <-  clust2[-split,]
names(traindf)
#Model with all attributes
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
glmfit1=train(Made.Donation.in.March.2007~.,data=traindf,  method = "glm", family = "binomial", trControl = ctrl, metric = "ROC")
summary(glmfit1)

#Model without Total Volume Donated
# Example of Logisic Regression algorithms
names(traindf)
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
logit <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "glm", family = "binomial", trControl = ctrl, metric = "ROC")
summary(logit)
#Finding confusion Matrix for Trainingg Data

logitProbs1 <- predict(logit, newdata=traindf, type="prob")[,1]
logitClasses1 <- predict(logit, newdata=traindf)
confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve for TRAININFng Data

rocCurve1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = logitProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurve1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Logit"), fill = c("red"))
auc(rocCurve1)

#Finding confusion Matrix for Testing Data

logitProbs <- predict(logit, newdata=testdf, type="prob")[,1]
logitClasses <- predict(logit, newdata=testdf)
confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve for Testing Data

rocCurve <- roc (response = testdf$Made.Donation.in.March.2007, predictor = logitProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurve, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Logit"), fill = c("red"))
auc(rocCurve)


#Bagging

bagmod=train(Made.Donation.in.March.2007~.,data=traindf,method="treebag",trControl=trainControl(method="cv",number=5))
#inding consusion matrix on training data
bagProbs1 <- predict(bagmod, newdata=traindf, type="prob")[,1]
bagClasses1 <- predict(bagmod, newdata=traindf)
confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on training data
rocCurvebag1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = bagProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvebag1, le1gacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Bagging"), fill = c("red"))
auc(rocCurvebag1)

#inding consusion matrix on testing data
bagProbs <- predict(bagmod, newdata=testdf, type="prob")[,1]
bagClasses <- predict(bagmod, newdata=testdf)
confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvebag <- roc (response = testdf$Made.Donation.in.March.2007, predictor = bagProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvebag, le1gacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Bagging"), fill = c("red"))
auc(rocCurvebag)

#Boosting

boostmod=train(Made.Donation.in.March.2007~.,data=traindf,method="gbm",verbose=F,trControl=trainControl(method="cv",number=60))
#Creating confusion matrix for training data
boostProbs1 <- predict(boostmod, newdata=traindf, type="prob")[,1]
boostClasses1 <- predict(boostmod, newdata=traindf)
confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve for training data
rocCurveboost1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = boostProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurveboost1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Boosting"), fill = c("red"))
auc(rocCurveboost1)
#Creating confusion matrix for testing data
boostProbs <- predict(boostmod, newdata=testdf, type="prob")[,1]
boostClasses <- predict(boostmod, newdata=testdf)
confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve for testing data
rocCurveboost <- roc (response = testdf$Made.Donation.in.March.2007, predictor = boostProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurveboost, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Boosting"), fill = c("red"))
auc(rocCurveboost)
#RandomForest 

RFmod=train(Made.Donation.in.March.2007~.,data=traindf,method="rf",importance=T,trControl=trainControl(method="cv",number=50))
#finding confusion matrix for training data
rfProbs1 <- predict(RFmod, newdata=traindf, type="prob")[,1]
rfClasses1 <- predict(RFmod, newdata=traindf)
confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on training data
rocCurverf1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = rfProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurverf1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Random Forest"), fill = c("red"))
auc(rocCurverf1)
#finding confusion matrix for testing data
rfProbs <- predict(RFmod, newdata=testdf, type="prob")[,1]
rfClasses <- predict(RFmod, newdata=testdf)
confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurverf <- roc (response = testdf$Made.Donation.in.March.2007, predictor = rfProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurverf, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Random Forest"), fill = c("red"))
auc(rocCurverf)
#n fold cross validation
# define training control

# 10-fold CV
ctrl <- trainControl(method = "cv", number=10,classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
CVmod <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "nb",tuneLength =10,family = "binomial", trControl = ctrl, metric = "ROC")

# summarize results
summary(CVmod)
#Predict on training data
cvProbs1 <- predict(CVmod, newdata=traindf, type="prob")[,1]
cvClasses1 <- predict(CVmod, newdata=traindf)
confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve ib training data

rocCurvecv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = cvProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("10 Fold Cross validation"), fill = c("red"))
auc(rocCurvecv1)
#Predict on testing data
cvProbs <- predict(CVmod, newdata=testdf, type="prob")[,1]
cvClasses <- predict(CVmod, newdata=testdf)
confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve ib testing data

rocCurvecv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = cvProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("10 Fold Cross validation"), fill = c("red"))
auc(rocCurvecv)
#Logitboost
library(caTools)
ctrl <- trainControl(classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
logitboostmod <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "LogitBoost",family = "binomial", tuneLength=50,trControl = ctrl, metric = "ROC")

# summarize results
summary(logitboostmod)
#Predict on training data
lbProbs1 <- predict(logitboostmod, newdata=traindf, type="prob")[,1]
lbClasses1 <- predict(logitboostmod, newdata=traindf)
confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelb1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = lbProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelb1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LogitBoost"), fill = c("red"))
auc(rocCurvelb1)
#Predict on testing data
lbProbs <- predict(logitboostmod, newdata=testdf, type="prob")[,1]
lbClasses <- predict(logitboostmod, newdata=testdf)
confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelb <- roc (response = testdf$Made.Donation.in.March.2007, predictor = lbProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelb, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LogitBoost"), fill = c("red"))
auc(rocCurvelb)

#LDA
ctrl <- trainControl(classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
ldamod <- train(Made.Donation.in.March.2007~.,data = traindf, method = "lda",family = "binomial",trControl = ctrl, metric = "ROC")
ldamod
traindf$Total.Volume.Donated..c.c..=NULL
# summarize results
summary(ldamod)
#Predict on training data
ldaProbs1 <- predict(ldamod, newdata=traindf, type="prob")[,1]
ldaClasses1 <- predict(ldamod, newdata=traindf)
confusionMatrix(data=ldaClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelda1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = ldaProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelda1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LDA"), fill = c("red"))
auc(rocCurvelda1)
#Predict on testing data
ldaProbs <- predict(ldamod, newdata=testdf, type="prob")[,1]
ldaClasses <- predict(ldamod, newdata=testdf)
confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvelda <- roc (response = testdf$Made.Donation.in.March.2007, predictor = ldaProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelda, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LDA"), fill = c("red"))
auc(rocCurvelda)

#svm RBF
# Training SVM Models
library(caret)
library(dplyr)         # Used by caret
library(kernlab)       # support vector machine 
library(pROC)	       # plot the ROC curves
trainX=traindf[,1:3]
names(trainX)
# Setup for cross validation
ctrl <- trainControl(method="repeatedcv",   # 10fold cross validation
                     repeats=5,		    # do 5 repititions of cv
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE)
svcvmod <- train(x=trainX,
                 y= traindf$Made.Donation.in.March.2007,
                 method = "svmRadial",   # Radial kernel
                 tuneLength = 9,					# 9 values of the cost function
                 preProc = c("center","scale"),  # Center and scale data
                 metric="ROC",
                 trControl=ctrl)
svcvmod
#Predict on training data
svcvProbs1 <- predict(svcvmod, newdata=trainX, type="prob")[,1]
svcvClasses1 <- predict(svcvmod, newdata=trainX)
confusionMatrix(data=svcvClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvesvcv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = svcvProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesvcv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesvcv1)
#Predict on testing data
names(testdf)
testdf$Total.Volume.Donated..c.c..=NULL
testX=testdf[,1:3]
names(testX)
svcvProbs <- predict(svcvmod, newdata=testX, type="prob")[,1]
svcvClasses <- predict(svcvmod, newdata=testX)
confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvesvcv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = svcvProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesvcv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesvcv)
#SVM wihout crossvalidation
ctrl <- trainControl(summaryFunction=twoClassSummary,	classProbs=TRUE)
svmod1 <- train(x=trainX,
                y= traindf$Made.Donation.in.March.2007,
                method = "svmRadial",   # Radial kernel
                tuneLength = 9,					# 9 values of the cost function
                preProc = c("center","scale"),  # Center and scale data
                metric="ROC",
                trControl=ctrl)
svmod1

#Predict on training data
svProbs1 <- predict(svmod1, newdata=trainX, type="prob")[,1]
svClasses1 <- predict(svmod1, newdata=trainX)
confusionMatrix(data=svClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvesv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = svProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesv1)
#Predict on testing data
svProbs <- predict(svmod1, newdata=testX, type="prob")[,1]
svClasses <- predict(svmod1, newdata=testX)
confusionMatrix(data=svClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvesv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = svProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesv)

#


# plot ROC curves for training 
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve1, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Training Data")
lines(rocCurvebag1, col="blue")
lines(rocCurveboost1, col="orange")
lines(rocCurverf1, col="light green")
lines(rocCurvecv1,col="yellow")
lines(rocCurvelb1,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))
# plot ROC curves for testing
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Testing Data")
lines(rocCurvebag, col="blue")
lines(rocCurveboost, col="orange")
lines(rocCurverf, col="light green")
lines(rocCurvecv,col="yellow")
lines(rocCurvelb,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))
#ANN MLP
#install.packages("RSNNS")
library(RSNNS)
ctrl <- trainC1
control(summaryFunction=twoClassSummary,	classProbs=TRUE)
mlpmod1 <- train(Made.Donation.in.March.2007~.,
                 traindf,
                 method = "mlp",
                 preProc = c("center","scale"),  # Center and scale data
                 metric="ROC",
                 trControl=ctrl)
mlpmod1

#Predict on training data
mlpProbs1 <- predict(mlpmod1, newdata=trainX, type="prob")[,1]
mlpClasses1 <- predict(mlpmod1, newdata=trainX)
confusionMatrix(mlpClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvemlp1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = mlpProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvemlp1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("MLP"), fill = c("red"))
auc(rocCurvemlp1)
#Predict on testing data
mlpProbs <- predict(mlpmod1, newdata=testX, type="prob")[,1]
mlpClasses <- predict(mlpmod1, newdata=testX)
confusionMatrix(mlpClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvemlp <- roc (response = testdf$Made.Donation.in.March.2007, predictor = mlpProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvemlp, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("MLP"), fill = c("red"))
auc(rocCurvemlp)

#PNN


#c5.0
install.packages("C50")
library(C50)
library(caret)
traindf
names(traindf)
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
#c5mod1 = train(Made.Donation.in.March.2007~.,trainmethod = "C5.0", preProc = c("center","scale"),metric="ROC")
c5mod1 <- train(Made.Donation.in.March.2007~.,
                traindf,
                method = "C5.0",
                preProc = c("center","scale"),  # Center and scale data
                metric="ROC",
                trControl=ctrl)

#Predict on training data
C5Probs1 <- predict(c5mod1, newdata=traindf, type="prob")[,1]
c5Classes1 <- predict(c5mod1, newdata=traindf)
confusionMatrix(c5Classes1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvec51 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = C5Probs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvec51, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("C5"), fill = c("red"))
auc(rocCurvec51)
#Predict on testing data
textX=testdf[,1:3]
textX
c5Probs <- predict(c5mod1, newdata=testdf, type="prob")[,1]
c5Classes <- predict(c5mod1, newdata=testdf)
confusionMatrix(c5Classes, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvec5 <- roc (response = testdf$Made.Donation.in.March.2007, predictor = c5Probs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvec5, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("C5"), fill = c("red"))
auc(rocCurvec5)
#CART
library(caret)
library(rpart)
cartmod1 <- train(Made.Donation.in.March.2007~.,
                  traindf,
                  method = "rpart",
                  preProc = c("center","scale"),  # Center and scale data
                  metric="ROC",
                  trControl=ctrl)
summary(cartmod1)

#Predict on training data
cartProbs1 <- predict(cartmod1, newdata=traindf, type="prob")[,1]
cartClasses1 <- predict(cartmod1, newdata=traindf)
confusionMatrix(cartClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on traininging data
library(pROC)

rocCurvecart1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = cartProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecart1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Cart"), fill = c("red"))
auc(rocCurvecart1)
#Predict on testing data
cartProbs <- predict(cartmod1, newdata=testdf, type="prob")[,1]
cartClasses <- predict(cartmod1, newdata=testdf)
confusionMatrix(cartClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvecart <- roc (response = testdf$Made.Donation.in.March.2007, predictor = cartProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecart, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("CART"), fill = c("red"))
auc(rocCurvecart)

# plot ROC curves for training 
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve1, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Training Data")
lines(rocCurvebag1, col="blue")
lines(rocCurveboost1, col="orange")
lines(rocCurverf1, col="light green")
lines(rocCurvecv1,col="yellow")
lines(rocCurvelb1,col="dark green")
lines(rocCurvelda1,col="light blue")
lines(rocCurvesvcv1,col="brown")
lines(rocCurvesv1,col="black")
lines(rocCurvemlp1,col="dark grey")
lines(rocCurvecart1,col="pink")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.5
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost", "LDA","SVM (RBF)"
                  ,"SVM","ANN MLP", "CART")
       , fill=c("red","blue","orange","light green","yellow","dark green","light blue","brown","black"
                ,"dark grey","pink"))
# plot ROC curves for testing
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Testing Data")
lines(rocCurvebag, col="blue")
lines(rocCurveboost, col="orange")
lines(rocCurverf, col="light green")
lines(rocCurvecv,col="yellow")
lines(rocCurvelb,col="dark green")
lines(rocCurvelda,col="light blue")
lines(rocCurvesvcv,col="brown")
lines(rocCurvesv,col="black")
lines(rocCurvemlp,col="grey")
lines(rocCurvecart,col="pink")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.5
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost", "LDA","SVM (RBF)"
                  ,"SVM","ANN MLP", "CART")
       , fill=c("red","blue","orange","light green","yellow","dark green","light blue","brown","black","grey","pink"))


#Storing Results in same Matrix fir Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve)[1])
m2 = cbind(confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag)[1])
m3 = cbind(confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost)[1])
m4 = cbind(confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf)[1])
m5 = cbind(confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv)[1])
m6 = cbind(confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m7 = cbind(confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m8 = cbind(confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m9 = cbind(confusionMatrix(data=svClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m10 = cbind(confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
m11 = cbind(confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
results <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results) <- models
names(results) <- c(stats)
results


#Storing Results in same Matrix for Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve1)[1])
m2 = cbind(confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag1)[1])
m3 = cbind(confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost1)[1])
m4 = cbind(confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf1)[1])
m5 = cbind(confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv1)[1])
m6 = cbind(confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb1)[1])
m7 = cbind(confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m8 = cbind(confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m9 = cbind(confusionMatrix(data=svClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m10 = cbind(confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
m11 = cbind(confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
results1 <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results1) <- models
names(results1) <- c(stats)
results1




#Run model on cluster3
clust3=subset(train1,train1$Cluster==3)
names(clust3)
#Divide the dataset into training and testing and run each model
library(caret)
library(plyr)
library(ipred)
library(e1071)
library(pROC)
library(gbm)
library(survival)
library(splines)
library(parallel)
library(randomForest)
library(MASS)
library(e1071)
summary(clust3)
clust3$x=NULL
clust3$Total.Volume.Donated..c.c..= NULL
clust3$Cluster=NULL

# split data set into training and testinG
set.seed(2016)
split <- createDataPartition(clust3$Made.Donation.in.March.2007, p=0.7, list=F)
traindf <- clust3[split,]
testdf <-  clust3[-split,]
names(traindf)
#Model with all attributes
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
glmfit1=train(Made.Donation.in.March.2007~.,data=traindf,  method = "glm", family = "binomial", trControl = ctrl, metric = "ROC")
summary(glmfit1)

#Model without Total Volume Donated
# Example of Logisic Regression algorithms
names(traindf)
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
logit <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "glm", family = "binomial", trControl = ctrl, metric = "ROC")
summary(logit)
#Finding confusion Matrix for Trainingg Data

logitProbs1 <- predict(logit, newdata=traindf, type="prob")[,1]
logitClasses1 <- predict(logit, newdata=traindf)
confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve for Testing Data

rocCurve1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = logitProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurve1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Logit"), fill = c("red"))
auc(rocCurve1)

#Finding confusion Matrix for Testing Data

logitProbs <- predict(logit, newdata=testdf, type="prob")[,1]
logitClasses <- predict(logit, newdata=testdf)
confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve for Testing Data

rocCurve <- roc (response = testdf$Made.Donation.in.March.2007, predictor = logitProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurve, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Logit"), fill = c("red"))
auc(rocCurve)


#Bagging

bagmod=train(Made.Donation.in.March.2007~.,data=traindf,method="treebag",trControl=trainControl(method="cv",number=5))
#inding consusion matrix on training data
bagProbs1 <- predict(bagmod, newdata=traindf, type="prob")[,1]
bagClasses1 <- predict(bagmod, newdata=traindf)
confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on training data
rocCurvebag1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = bagProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvebag1, le1gacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Bagging"), fill = c("red"))
auc(rocCurvebag1)

#inding consusion matrix on testing data
bagProbs <- predict(bagmod, newdata=testdf, type="prob")[,1]
bagClasses <- predict(bagmod, newdata=testdf)
confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvebag <- roc (response = testdf$Made.Donation.in.March.2007, predictor = bagProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvebag, le1gacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Bagging"), fill = c("red"))
auc(rocCurvebag)

#Boosting

boostmod=train(Made.Donation.in.March.2007~.,data=traindf,method="gbm",verbose=F,trControl=trainControl(method="cv",number=60))
#Creating confusion matrix for training data
boostProbs1 <- predict(boostmod, newdata=traindf, type="prob")[,1]
boostClasses1 <- predict(boostmod, newdata=traindf)
confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve for training data
rocCurveboost1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = boostProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurveboost1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Boosting"), fill = c("red"))
auc(rocCurveboost1)
#Creating confusion matrix for testing data
boostProbs <- predict(boostmod, newdata=testdf, type="prob")[,1]
boostClasses <- predict(boostmod, newdata=testdf)
confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve for testing data
rocCurveboost <- roc (response = testdf$Made.Donation.in.March.2007, predictor = boostProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurveboost, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Boosting"), fill = c("red"))
auc(rocCurveboost)
#RandomForest 

RFmod=train(Made.Donation.in.March.2007~.,data=traindf,method="rf",importance=T,trControl=trainControl(method="cv",number=50))
#finding confusion matrix for training data
rfProbs1 <- predict(RFmod, newdata=traindf, type="prob")[,1]
rfClasses1 <- predict(RFmod, newdata=traindf)
confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on training data
rocCurverf1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = rfProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurverf1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Random Forest"), fill = c("red"))
auc(rocCurverf1)
#finding confusion matrix for testing data
rfProbs <- predict(RFmod, newdata=testdf, type="prob")[,1]
rfClasses <- predict(RFmod, newdata=testdf)
confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurverf <- roc (response = testdf$Made.Donation.in.March.2007, predictor = rfProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurverf, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Random Forest"), fill = c("red"))
auc(rocCurverf)
#n fold cross validation
# define training control

# 10-fold CV
ctrl <- trainControl(method = "cv", number=10,classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
CVmod <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "nb",tuneLength =10,family = "binomial", trControl = ctrl, metric = "ROC")

# summarize results
summary(CVmod)
#Predict on training data
cvProbs1 <- predict(CVmod, newdata=traindf, type="prob")[,1]
cvClasses1 <- predict(CVmod, newdata=traindf)
confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve ib training data

rocCurvecv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = cvProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("10 Fold Cross validation"), fill = c("red"))
auc(rocCurvecv1)
#Predict on testing data
cvProbs <- predict(CVmod, newdata=testdf, type="prob")[,1]
cvClasses <- predict(CVmod, newdata=testdf)
confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve ib testing data

rocCurvecv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = cvProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("10 Fold Cross validation"), fill = c("red"))
auc(rocCurvecv)
#Logitboost
library(caTools)
ctrl <- trainControl(classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
logitboostmod <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "LogitBoost",family = "binomial", tuneLength=50,trControl = ctrl, metric = "ROC")

# summarize results
summary(logitboostmod)
#Predict on training data
lbProbs1 <- predict(logitboostmod, newdata=traindf, type="prob")[,1]
lbClasses1 <- predict(logitboostmod, newdata=traindf)
confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelb1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = lbProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelb1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LogitBoost"), fill = c("red"))
auc(rocCurvelb1)
#Predict on testing data
lbProbs <- predict(logitboostmod, newdata=testdf, type="prob")[,1]
lbClasses <- predict(logitboostmod, newdata=testdf)
confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelb <- roc (response = testdf$Made.Donation.in.March.2007, predictor = lbProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelb, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LogitBoost"), fill = c("red"))
auc(rocCurvelb)

# plot ROC curves for training 
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve1, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Training Data")
lines(rocCurvebag1, col="blue")
lines(rocCurveboost1, col="orange")
lines(rocCurverf1, col="light green")
lines(rocCurvecv1,col="yellow")
lines(rocCurvelb1,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))
# plot ROC curves for testing
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Testing Data")
lines(rocCurvebag, col="blue")
lines(rocCurveboost, col="orange")
lines(rocCurverf, col="light green")
lines(rocCurvecv,col="yellow")
lines(rocCurvelb,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))


#Storing Results in same Matrix fir Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve)[1])
m2 = cbind(confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag)[1])
m3 = cbind(confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost)[1])
m4 = cbind(confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf)[1])
m5 = cbind(confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv)[1])
m6 = cbind(confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
results <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results) <- models
names(results) <- c(stats)
results


#Storing Results in same Matrix for Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve1)[1])
m2 = cbind(confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag1)[1])
m3 = cbind(confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost1)[1])
m4 = cbind(confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf1)[1])
m5 = cbind(confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv1)[1])
m6 = cbind(confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb1)[1])
results1 <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results1) <- models
names(results1) <- c(stats)
results1

#LDA
ctrl <- trainControl(classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
ldamod <- train(Made.Donation.in.March.2007~.,data = traindf, method = "lda",family = "binomial",trControl = ctrl, metric = "ROC")
ldamod
traindf$Total.Volume.Donated..c.c..=NULL
# summarize results
summary(ldamod)
#Predict on training data
ldaProbs1 <- predict(ldamod, newdata=traindf, type="prob")[,1]
ldaClasses1 <- predict(ldamod, newdata=traindf)
confusionMatrix(data=ldaClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelda1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = ldaProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelda1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LDA"), fill = c("red"))
auc(rocCurvelda1)
#Predict on testing data
ldaProbs <- predict(ldamod, newdata=testdf, type="prob")[,1]
ldaClasses <- predict(ldamod, newdata=testdf)
confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvelda <- roc (response = testdf$Made.Donation.in.March.2007, predictor = ldaProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelda, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LDA"), fill = c("red"))
auc(rocCurvelda)

#svm RBF
# Training SVM Models
library(caret)
library(dplyr)         # Used by caret
library(kernlab)       # support vector machine 
library(pROC)	       # plot the ROC curves
trainX=traindf[,1:3]
names(trainX)
# Setup for cross validation
ctrl <- trainControl(method="repeatedcv",   # 10fold cross validation
                     repeats=5,		    # do 5 repititions of cv
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE)
svcvmod <- train(x=trainX,
                 y= traindf$Made.Donation.in.March.2007,
                 method = "svmRadial",   # Radial kernel
                 tuneLength = 9,					# 9 values of the cost function
                 preProc = c("center","scale"),  # Center and scale data
                 metric="ROC",
                 trControl=ctrl)
svcvmod
#Predict on training data
svcvProbs1 <- predict(svcvmod, newdata=trainX, type="prob")[,1]
svcvClasses1 <- predict(svcvmod, newdata=trainX)
confusionMatrix(data=svcvClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvesvcv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = svcvProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesvcv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesvcv1)
#Predict on testing data
names(testdf)
testdf$Total.Volume.Donated..c.c..=NULL
testX=testdf[,1:3]
names(testX)
svcvProbs <- predict(svcvmod, newdata=testX, type="prob")[,1]
svcvClasses <- predict(svcvmod, newdata=testX)
confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvesvcv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = svcvProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesvcv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesvcv)
#SVM wihout crossvalidation
ctrl <- trainControl(summaryFunction=twoClassSummary,	classProbs=TRUE)
svmod1 <- train(x=trainX,
                y= traindf$Made.Donation.in.March.2007,
                method = "svmRadial",   # Radial kernel
                tuneLength = 9,					# 9 values of the cost function
                preProc = c("center","scale"),  # Center and scale data
                metric="ROC",
                trControl=ctrl)
svmod1

#Predict on training data
svProbs1 <- predict(svmod1, newdata=trainX, type="prob")[,1]
svClasses1 <- predict(svmod1, newdata=trainX)
confusionMatrix(data=svClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvesv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = svProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesv1)
#Predict on testing data
svProbs <- predict(svmod1, newdata=testX, type="prob")[,1]
svClasses <- predict(svmod1, newdata=testX)
confusionMatrix(data=svClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvesv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = svProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesv)

#


# plot ROC curves for training 
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve1, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Training Data")
lines(rocCurvebag1, col="blue")
lines(rocCurveboost1, col="orange")
lines(rocCurverf1, col="light green")
lines(rocCurvecv1,col="yellow")
lines(rocCurvelb1,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))
# plot ROC curves for testing
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Testing Data")
lines(rocCurvebag, col="blue")
lines(rocCurveboost, col="orange")
lines(rocCurverf, col="light green")
lines(rocCurvecv,col="yellow")
lines(rocCurvelb,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))
#ANN MLP
#install.packages("RSNNS")
library(RSNNS)
ctrl <- trainC1
control(summaryFunction=twoClassSummary,	classProbs=TRUE)
mlpmod1 <- train(Made.Donation.in.March.2007~.,
                 traindf,
                 method = "mlp",
                 preProc = c("center","scale"),  # Center and scale data
                 metric="ROC",
                 trControl=ctrl)
mlpmod1

#Predict on training data
mlpProbs1 <- predict(mlpmod1, newdata=trainX, type="prob")[,1]
mlpClasses1 <- predict(mlpmod1, newdata=trainX)
confusionMatrix(mlpClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvemlp1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = mlpProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvemlp1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("MLP"), fill = c("red"))
auc(rocCurvemlp1)
#Predict on testing data
mlpProbs <- predict(mlpmod1, newdata=testX, type="prob")[,1]
mlpClasses <- predict(mlpmod1, newdata=testX)
confusionMatrix(mlpClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvemlp <- roc (response = testdf$Made.Donation.in.March.2007, predictor = mlpProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvemlp, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("MLP"), fill = c("red"))
auc(rocCurvemlp)

#PNN


#c5.0
install.packages("C50")
library(C50)
library(caret)
traindf
names(traindf)
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
#c5mod1 = train(Made.Donation.in.March.2007~.,trainmethod = "C5.0", preProc = c("center","scale"),metric="ROC")
c5mod1 <- train(Made.Donation.in.March.2007~.,
                traindf,
                method = "C5.0",
                preProc = c("center","scale"),  # Center and scale data
                metric="ROC",
                trControl=ctrl)

#Predict on training data
C5Probs1 <- predict(c5mod1, newdata=traindf, type="prob")[,1]
c5Classes1 <- predict(c5mod1, newdata=traindf)
confusionMatrix(c5Classes1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvec51 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = C5Probs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvec51, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("C5"), fill = c("red"))
auc(rocCurvec51)
#Predict on testing data
textX=testdf[,1:3]
textX
c5Probs <- predict(c5mod1, newdata=testdf, type="prob")[,1]
c5Classes <- predict(c5mod1, newdata=testdf)
confusionMatrix(c5Classes, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvec5 <- roc (response = testdf$Made.Donation.in.March.2007, predictor = c5Probs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvec5, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("C5"), fill = c("red"))
auc(rocCurvec5)
#CART
library(caret)
library(rpart)
cartmod1 <- train(Made.Donation.in.March.2007~.,
                  traindf,
                  method = "rpart",
                  preProc = c("center","scale"),  # Center and scale data
                  metric="ROC",
                  trControl=ctrl)
summary(cartmod1)

#Predict on training data
cartProbs1 <- predict(cartmod1, newdata=traindf, type="prob")[,1]
cartClasses1 <- predict(cartmod1, newdata=traindf)
confusionMatrix(cartClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on traininging data
library(pROC)

rocCurvecart1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = cartProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecart1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Cart"), fill = c("red"))
auc(rocCurvecart1)
#Predict on testing data
cartProbs <- predict(cartmod1, newdata=testdf, type="prob")[,1]
cartClasses <- predict(cartmod1, newdata=testdf)
confusionMatrix(cartClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvecart <- roc (response = testdf$Made.Donation.in.March.2007, predictor = cartProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecart, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("CART"), fill = c("red"))
auc(rocCurvecart)

# plot ROC curves for training 
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve1, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Training Data")
lines(rocCurvebag1, col="blue")
lines(rocCurveboost1, col="orange")
lines(rocCurverf1, col="light green")
lines(rocCurvecv1,col="yellow")
lines(rocCurvelb1,col="dark green")
lines(rocCurvelda1,col="light blue")
lines(rocCurvesvcv1,col="brown")
lines(rocCurvesv1,col="black")
lines(rocCurvemlp1,col="dark grey")
lines(rocCurvecart1,col="pink")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.5
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost", "LDA","SVM (RBF)"
                  ,"SVM","ANN MLP", "CART")
       , fill=c("red","blue","orange","light green","yellow","dark green","light blue","brown","black"
                ,"dark grey","pink"))
# plot ROC curves for testing
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Testing Data")
lines(rocCurvebag, col="blue")
lines(rocCurveboost, col="orange")
lines(rocCurverf, col="light green")
lines(rocCurvecv,col="yellow")
lines(rocCurvelb,col="dark green")
lines(rocCurvelda,col="light blue")
lines(rocCurvesvcv,col="brown")
lines(rocCurvesv,col="black")
lines(rocCurvemlp,col="grey")
lines(rocCurvecart,col="pink")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.5
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost", "LDA","SVM (RBF)"
                  ,"SVM","ANN MLP", "CART")
       , fill=c("red","blue","orange","light green","yellow","dark green","light blue","brown","black","grey","pink"))


#Storing Results in same Matrix fir Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve)[1])
m2 = cbind(confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag)[1])
m3 = cbind(confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost)[1])
m4 = cbind(confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf)[1])
m5 = cbind(confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv)[1])
m6 = cbind(confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m7 = cbind(confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m8 = cbind(confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m9 = cbind(confusionMatrix(data=svClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m10 = cbind(confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
m11 = cbind(confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
results <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results) <- models
names(results) <- c(stats)
results


#Storing Results in same Matrix for Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve1)[1])
m2 = cbind(confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag1)[1])
m3 = cbind(confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost1)[1])
m4 = cbind(confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf1)[1])
m5 = cbind(confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv1)[1])
m6 = cbind(confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb1)[1])
m7 = cbind(confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m8 = cbind(confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m9 = cbind(confusionMatrix(data=svClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m10 = cbind(confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
m11 = cbind(confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
results1 <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results1) <- models
names(results1) <- c(stats)
results1


#Run model on cluster4
clust4=subset(train1,train1$Cluster==4)
names(clust4)
#Divide the dataset into training and testing and run each model
library(caret)
library(plyr)
library(ipred)
library(e1071)
library(pROC)
library(gbm)
library(survival)
library(splines)
library(parallel)
library(randomForest)
library(MASS)
library(e1071)
summary(clust4)
clust4$x=NULL
clust4$Total.Volume.Donated..c.c..= NULL
clust4$Cluster=NULL

# split data set into training and testinG
set.seed(2016)
split <- createDataPartition(clust4$Made.Donation.in.March.2007, p=0.7, list=F)
traindf <- clust4[split,]
testdf <-  clust4[-split,]
names(traindf)
#Model with all attributes
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
glmfit1=train(Made.Donation.in.March.2007~.,data=traindf,  method = "glm", family = "binomial", trControl = ctrl, metric = "ROC")
summary(glmfit1)

#Model without Total Volume Donated
# Example of Logisic Regression algorithms
names(traindf)
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
logit <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "glm", family = "binomial", trControl = ctrl, metric = "ROC")
summary(logit)
#Finding confusion Matrix for Trainingg Data

logitProbs1 <- predict(logit, newdata=traindf, type="prob")[,1]
logitClasses1 <- predict(logit, newdata=traindf)
confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve for Testing Data

rocCurve1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = logitProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurve1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Logit"), fill = c("red"))
auc(rocCurve1)

#Finding confusion Matrix for Testing Data

logitProbs <- predict(logit, newdata=testdf, type="prob")[,1]
logitClasses <- predict(logit, newdata=testdf)
confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve for Testing Data

rocCurve <- roc (response = testdf$Made.Donation.in.March.2007, predictor = logitProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurve, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Logit"), fill = c("red"))
auc(rocCurve)


#Bagging

bagmod=train(Made.Donation.in.March.2007~.,data=traindf,method="treebag",trControl=trainControl(method="cv",number=5))
#inding consusion matrix on training data
bagProbs1 <- predict(bagmod, newdata=traindf, type="prob")[,1]
bagClasses1 <- predict(bagmod, newdata=traindf)
confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on training data
rocCurvebag1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = bagProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvebag1, le1gacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Bagging"), fill = c("red"))
auc(rocCurvebag1)

#inding consusion matrix on testing data
bagProbs <- predict(bagmod, newdata=testdf, type="prob")[,1]
bagClasses <- predict(bagmod, newdata=testdf)
confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvebag <- roc (response = testdf$Made.Donation.in.March.2007, predictor = bagProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvebag, le1gacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Bagging"), fill = c("red"))
auc(rocCurvebag)

#Boosting

boostmod=train(Made.Donation.in.March.2007~.,data=traindf,method="gbm",verbose=F,trControl=trainControl(method="cv",number=60))
#Creating confusion matrix for training data
boostProbs1 <- predict(boostmod, newdata=traindf, type="prob")[,1]
boostClasses1 <- predict(boostmod, newdata=traindf)
confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve for training data
rocCurveboost1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = boostProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurveboost1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Boosting"), fill = c("red"))
auc(rocCurveboost1)
#Creating confusion matrix for testing data
boostProbs <- predict(boostmod, newdata=testdf, type="prob")[,1]
boostClasses <- predict(boostmod, newdata=testdf)
confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve for testing data
rocCurveboost <- roc (response = testdf$Made.Donation.in.March.2007, predictor = boostProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurveboost, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Boosting"), fill = c("red"))
auc(rocCurveboost)
#RandomForest 

RFmod=train(Made.Donation.in.March.2007~.,data=traindf,method="rf",importance=T,trControl=trainControl(method="cv",number=50))
#finding confusion matrix for training data
rfProbs1 <- predict(RFmod, newdata=traindf, type="prob")[,1]
rfClasses1 <- predict(RFmod, newdata=traindf)
confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on training data
rocCurverf1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = rfProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurverf1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Random Forest"), fill = c("red"))
auc(rocCurverf1)
#finding confusion matrix for testing data
rfProbs <- predict(RFmod, newdata=testdf, type="prob")[,1]
rfClasses <- predict(RFmod, newdata=testdf)
confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurverf <- roc (response = testdf$Made.Donation.in.March.2007, predictor = rfProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurverf, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Random Forest"), fill = c("red"))
auc(rocCurverf)
#n fold cross validation
# define training control

# 10-fold CV
ctrl <- trainControl(method = "cv", number=10,classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
CVmod <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "nb",tuneLength =10,family = "binomial", trControl = ctrl, metric = "ROC")

# summarize results
summary(CVmod)
#Predict on training data
cvProbs1 <- predict(CVmod, newdata=traindf, type="prob")[,1]
cvClasses1 <- predict(CVmod, newdata=traindf)
confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve ib training data

rocCurvecv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = cvProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("10 Fold Cross validation"), fill = c("red"))
auc(rocCurvecv1)
#Predict on testing data
cvProbs <- predict(CVmod, newdata=testdf, type="prob")[,1]
cvClasses <- predict(CVmod, newdata=testdf)
confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve ib testing data

rocCurvecv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = cvProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("10 Fold Cross validation"), fill = c("red"))
auc(rocCurvecv)
#Logitboost
library(caTools)
ctrl <- trainControl(classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
logitboostmod <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "LogitBoost",family = "binomial", tuneLength=50,trControl = ctrl, metric = "ROC")

# summarize results
summary(logitboostmod)
#Predict on training data
lbProbs1 <- predict(logitboostmod, newdata=traindf, type="prob")[,1]
lbClasses1 <- predict(logitboostmod, newdata=traindf)
confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelb1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = lbProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelb1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LogitBoost"), fill = c("red"))
auc(rocCurvelb1)
#Predict on testing data
lbProbs <- predict(logitboostmod, newdata=testdf, type="prob")[,1]
lbClasses <- predict(logitboostmod, newdata=testdf)
confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelb <- roc (response = testdf$Made.Donation.in.March.2007, predictor = lbProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelb, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LogitBoost"), fill = c("red"))
auc(rocCurvelb)

# plot ROC curves for training 
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve1, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Training Data")
lines(rocCurvebag1, col="blue")
lines(rocCurveboost1, col="orange")
lines(rocCurverf1, col="light green")
lines(rocCurvecv1,col="yellow")
lines(rocCurvelb1,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))
# plot ROC curves for testing
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Testing Data")
lines(rocCurvebag, col="blue")
lines(rocCurveboost, col="orange")
lines(rocCurverf, col="light green")
lines(rocCurvecv,col="yellow")
lines(rocCurvelb,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))


#Storing Results in same Matrix fir Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve)[1])
m2 = cbind(confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag)[1])
m3 = cbind(confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost)[1])
m4 = cbind(confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf)[1])
m5 = cbind(confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv)[1])
m6 = cbind(confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
results <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results) <- models
names(results) <- c(stats)
results

#Storing Results in same Matrix for Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve1)[1])
m2 = cbind(confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag1)[1])
m3 = cbind(confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost1)[1])
m4 = cbind(confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf1)[1])
m5 = cbind(confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv1)[1])
m6 = cbind(confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb1)[1])
results1 <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results1) <- models
names(results1) <- c(stats)
results1

#LDA
ctrl <- trainControl(classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
ldamod <- train(Made.Donation.in.March.2007~.,data = traindf, method = "lda",family = "binomial",trControl = ctrl, metric = "ROC")
ldamod
traindf$Total.Volume.Donated..c.c..=NULL
# summarize results
summary(ldamod)
#Predict on training data
ldaProbs1 <- predict(ldamod, newdata=traindf, type="prob")[,1]
ldaClasses1 <- predict(ldamod, newdata=traindf)
confusionMatrix(data=ldaClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelda1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = ldaProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelda1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LDA"), fill = c("red"))
auc(rocCurvelda1)
#Predict on testing data
ldaProbs <- predict(ldamod, newdata=testdf, type="prob")[,1]
ldaClasses <- predict(ldamod, newdata=testdf)
confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvelda <- roc (response = testdf$Made.Donation.in.March.2007, predictor = ldaProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelda, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LDA"), fill = c("red"))
auc(rocCurvelda)

#svm RBF
# Training SVM Models
library(caret)
library(dplyr)         # Used by caret
library(kernlab)       # support vector machine 
library(pROC)	       # plot the ROC curves
trainX=traindf[,1:3]
names(trainX)
# Setup for cross validation
ctrl <- trainControl(method="repeatedcv",   # 10fold cross validation
                     repeats=5,		    # do 5 repititions of cv
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE)
svcvmod <- train(x=trainX,
                 y= traindf$Made.Donation.in.March.2007,
                 method = "svmRadial",   # Radial kernel
                 tuneLength = 9,					# 9 values of the cost function
                 preProc = c("center","scale"),  # Center and scale data
                 metric="ROC",
                 trControl=ctrl)
svcvmod
#Predict on training data
svcvProbs1 <- predict(svcvmod, newdata=trainX, type="prob")[,1]
svcvClasses1 <- predict(svcvmod, newdata=trainX)
confusionMatrix(data=svcvClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvesvcv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = svcvProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesvcv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesvcv1)
#Predict on testing data
names(testdf)
testdf$Total.Volume.Donated..c.c..=NULL
testX=testdf[,1:3]
names(testX)
svcvProbs <- predict(svcvmod, newdata=testX, type="prob")[,1]
svcvClasses <- predict(svcvmod, newdata=testX)
confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvesvcv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = svcvProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesvcv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesvcv)
#SVM wihout crossvalidation
ctrl <- trainControl(summaryFunction=twoClassSummary,	classProbs=TRUE)
svmod1 <- train(x=trainX,
                y= traindf$Made.Donation.in.March.2007,
                method = "svmRadial",   # Radial kernel
                tuneLength = 9,					# 9 values of the cost function
                preProc = c("center","scale"),  # Center and scale data
                metric="ROC",
                trControl=ctrl)
svmod1

#Predict on training data
svProbs1 <- predict(svmod1, newdata=trainX, type="prob")[,1]
svClasses1 <- predict(svmod1, newdata=trainX)
confusionMatrix(data=svClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvesv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = svProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesv1)
#Predict on testing data
svProbs <- predict(svmod1, newdata=testX, type="prob")[,1]
svClasses <- predict(svmod1, newdata=testX)
confusionMatrix(data=svClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvesv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = svProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesv)

#


# plot ROC curves for training 
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve1, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Training Data")
lines(rocCurvebag1, col="blue")
lines(rocCurveboost1, col="orange")
lines(rocCurverf1, col="light green")
lines(rocCurvecv1,col="yellow")
lines(rocCurvelb1,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))
# plot ROC curves for testing
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Testing Data")
lines(rocCurvebag, col="blue")
lines(rocCurveboost, col="orange")
lines(rocCurverf, col="light green")
lines(rocCurvecv,col="yellow")
lines(rocCurvelb,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))
#ANN MLP
#install.packages("RSNNS")
library(RSNNS)
ctrl <- trainC1
control(summaryFunction=twoClassSummary,	classProbs=TRUE)
mlpmod1 <- train(Made.Donation.in.March.2007~.,
                 traindf,
                 method = "mlp",
                 preProc = c("center","scale"),  # Center and scale data
                 metric="ROC",
                 trControl=ctrl)
mlpmod1

#Predict on training data
mlpProbs1 <- predict(mlpmod1, newdata=trainX, type="prob")[,1]
mlpClasses1 <- predict(mlpmod1, newdata=trainX)
confusionMatrix(mlpClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvemlp1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = mlpProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvemlp1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("MLP"), fill = c("red"))
auc(rocCurvemlp1)
#Predict on testing data
mlpProbs <- predict(mlpmod1, newdata=testX, type="prob")[,1]
mlpClasses <- predict(mlpmod1, newdata=testX)
confusionMatrix(mlpClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvemlp <- roc (response = testdf$Made.Donation.in.March.2007, predictor = mlpProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvemlp, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("MLP"), fill = c("red"))
auc(rocCurvemlp)

#PNN


#c5.0
install.packages("C50")
library(C50)
library(caret)
traindf
names(traindf)
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
#c5mod1 = train(Made.Donation.in.March.2007~.,trainmethod = "C5.0", preProc = c("center","scale"),metric="ROC")
c5mod1 <- train(Made.Donation.in.March.2007~.,
                traindf,
                method = "C5.0",
                preProc = c("center","scale"),  # Center and scale data
                metric="ROC",
                trControl=ctrl)

#Predict on training data
C5Probs1 <- predict(c5mod1, newdata=traindf, type="prob")[,1]
c5Classes1 <- predict(c5mod1, newdata=traindf)
confusionMatrix(c5Classes1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvec51 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = C5Probs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvec51, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("C5"), fill = c("red"))
auc(rocCurvec51)
#Predict on testing data
textX=testdf[,1:3]
textX
c5Probs <- predict(c5mod1, newdata=testdf, type="prob")[,1]
c5Classes <- predict(c5mod1, newdata=testdf)
confusionMatrix(c5Classes, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvec5 <- roc (response = testdf$Made.Donation.in.March.2007, predictor = c5Probs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvec5, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("C5"), fill = c("red"))
auc(rocCurvec5)
#CART
library(caret)
library(rpart)
cartmod1 <- train(Made.Donation.in.March.2007~.,
                  traindf,
                  method = "rpart",
                  preProc = c("center","scale"),  # Center and scale data
                  metric="ROC",
                  trControl=ctrl)
summary(cartmod1)

#Predict on training data
cartProbs1 <- predict(cartmod1, newdata=traindf, type="prob")[,1]
cartClasses1 <- predict(cartmod1, newdata=traindf)
confusionMatrix(cartClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on traininging data
library(pROC)

rocCurvecart1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = cartProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecart1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Cart"), fill = c("red"))
auc(rocCurvecart1)
#Predict on testing data
cartProbs <- predict(cartmod1, newdata=testdf, type="prob")[,1]
cartClasses <- predict(cartmod1, newdata=testdf)
confusionMatrix(cartClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvecart <- roc (response = testdf$Made.Donation.in.March.2007, predictor = cartProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecart, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("CART"), fill = c("red"))
auc(rocCurvecart)

# plot ROC curves for training 
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve1, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Training Data")
lines(rocCurvebag1, col="blue")
lines(rocCurveboost1, col="orange")
lines(rocCurverf1, col="light green")
lines(rocCurvecv1,col="yellow")
lines(rocCurvelb1,col="dark green")
lines(rocCurvelda1,col="light blue")
lines(rocCurvesvcv1,col="brown")
lines(rocCurvesv1,col="black")
lines(rocCurvemlp1,col="dark grey")
lines(rocCurvecart1,col="pink")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.5
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost", "LDA","SVM (RBF)"
                  ,"SVM","ANN MLP", "CART")
       , fill=c("red","blue","orange","light green","yellow","dark green","light blue","brown","black"
                ,"dark grey","pink"))
# plot ROC curves for testing
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Testing Data")
lines(rocCurvebag, col="blue")
lines(rocCurveboost, col="orange")
lines(rocCurverf, col="light green")
lines(rocCurvecv,col="yellow")
lines(rocCurvelb,col="dark green")
lines(rocCurvelda,col="light blue")
lines(rocCurvesvcv,col="brown")
lines(rocCurvesv,col="black")
lines(rocCurvemlp,col="grey")
lines(rocCurvecart,col="pink")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.5
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost", "LDA","SVM (RBF)"
                  ,"SVM","ANN MLP", "CART")
       , fill=c("red","blue","orange","light green","yellow","dark green","light blue","brown","black","grey","pink"))


#Storing Results in same Matrix fir Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve)[1])
m2 = cbind(confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag)[1])
m3 = cbind(confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost)[1])
m4 = cbind(confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf)[1])
m5 = cbind(confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv)[1])
m6 = cbind(confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m7 = cbind(confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m8 = cbind(confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m9 = cbind(confusionMatrix(data=svClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m10 = cbind(confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
m11 = cbind(confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
results <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results) <- models
names(results) <- c(stats)
results


#Storing Results in same Matrix for Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve1)[1])
m2 = cbind(confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag1)[1])
m3 = cbind(confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost1)[1])
m4 = cbind(confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf1)[1])
m5 = cbind(confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv1)[1])
m6 = cbind(confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb1)[1])
m7 = cbind(confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m8 = cbind(confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m9 = cbind(confusionMatrix(data=svClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m10 = cbind(confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
m11 = cbind(confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
results1 <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results1) <- models
names(results1) <- c(stats)
results1


#Run model on cluster5
clust5=subset(train1,train1$Cluster==5)
names(clust5)
#Divide the dataset into training and testing and run each model
library(caret)
library(plyr)
library(ipred)
library(e1071)
library(pROC)
library(gbm)
library(survival)
library(splines)
library(parallel)
library(randomForest)
library(MASS)
library(e1071)
summary(clust4)
clust5$x=NULL
clust5$Total.Volume.Donated..c.c..= NULL
clust5$Cluster=NULL

# split data set into training and testinG
set.seed(2016)
split <- createDataPartition(clust5$Made.Donation.in.March.2007, p=0.7, list=F)
traindf <- clust5[split,]
testdf <-  clust5[-split,]
names(traindf)
#Model with all attributes
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
glmfit1=train(Made.Donation.in.March.2007~.,data=traindf,  method = "glm", family = "binomial", trControl = ctrl, metric = "ROC")
summary(glmfit1)

#Model without Total Volume Donated
# Example of Logisic Regression algorithms
names(traindf)
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
logit <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "glm", family = "binomial", trControl = ctrl, metric = "ROC")
summary(logit)
#Finding confusion Matrix for Trainingg Data

logitProbs1 <- predict(logit, newdata=traindf, type="prob")[,1]
logitClasses1 <- predict(logit, newdata=traindf)
confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve for Testing Data

rocCurve1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = logitProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurve1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Logit"), fill = c("red"))
auc(rocCurve1)

#Finding confusion Matrix for Testing Data

logitProbs <- predict(logit, newdata=testdf, type="prob")[,1]
logitClasses <- predict(logit, newdata=testdf)
confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve for Testing Data

rocCurve <- roc (response = testdf$Made.Donation.in.March.2007, predictor = logitProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurve, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Logit"), fill = c("red"))
auc(rocCurve)


#Bagging

bagmod=train(Made.Donation.in.March.2007~.,data=traindf,method="treebag",trControl=trainControl(method="cv",number=5))
#inding consusion matrix on training data
bagProbs1 <- predict(bagmod, newdata=traindf, type="prob")[,1]
bagClasses1 <- predict(bagmod, newdata=traindf)
confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on training data
rocCurvebag1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = bagProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvebag1, le1gacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Bagging"), fill = c("red"))
auc(rocCurvebag1)

#inding consusion matrix on testing data
bagProbs <- predict(bagmod, newdata=testdf, type="prob")[,1]
bagClasses <- predict(bagmod, newdata=testdf)
confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvebag <- roc (response = testdf$Made.Donation.in.March.2007, predictor = bagProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvebag, le1gacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Bagging"), fill = c("red"))
auc(rocCurvebag)

#Boosting

boostmod=train(Made.Donation.in.March.2007~.,data=traindf,method="gbm",verbose=F,trControl=trainControl(method="cv",number=60))
#Creating confusion matrix for training data
boostProbs1 <- predict(boostmod, newdata=traindf, type="prob")[,1]
boostClasses1 <- predict(boostmod, newdata=traindf)
confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve for training data
rocCurveboost1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = boostProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurveboost1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Boosting"), fill = c("red"))
auc(rocCurveboost1)
#Creating confusion matrix for testing data
boostProbs <- predict(boostmod, newdata=testdf, type="prob")[,1]
boostClasses <- predict(boostmod, newdata=testdf)
confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve for testing data
rocCurveboost <- roc (response = testdf$Made.Donation.in.March.2007, predictor = boostProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurveboost, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Boosting"), fill = c("red"))
auc(rocCurveboost)
#RandomForest 

RFmod=train(Made.Donation.in.March.2007~.,data=traindf,method="rf",importance=T,trControl=trainControl(method="cv",number=50))
#finding confusion matrix for training data
rfProbs1 <- predict(RFmod, newdata=traindf, type="prob")[,1]
rfClasses1 <- predict(RFmod, newdata=traindf)
confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on training data
rocCurverf1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = rfProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurverf1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Random Forest"), fill = c("red"))
auc(rocCurverf1)
#finding confusion matrix for testing data
rfProbs <- predict(RFmod, newdata=testdf, type="prob")[,1]
rfClasses <- predict(RFmod, newdata=testdf)
confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurverf <- roc (response = testdf$Made.Donation.in.March.2007, predictor = rfProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurverf, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Random Forest"), fill = c("red"))
auc(rocCurverf)
#n fold cross validation
# define training control

# 10-fold CV
ctrl <- trainControl(method = "cv", number=10,classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
CVmod <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "nb",tuneLength =10,family = "binomial", trControl = ctrl, metric = "ROC")

# summarize results
summary(CVmod)
#Predict on training data
cvProbs1 <- predict(CVmod, newdata=traindf, type="prob")[,1]
cvClasses1 <- predict(CVmod, newdata=traindf)
confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve ib training data

rocCurvecv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = cvProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("10 Fold Cross validation"), fill = c("red"))
auc(rocCurvecv1)
#Predict on testing data
cvProbs <- predict(CVmod, newdata=testdf, type="prob")[,1]
cvClasses <- predict(CVmod, newdata=testdf)
confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve ib testing data

rocCurvecv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = cvProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("10 Fold Cross validation"), fill = c("red"))
auc(rocCurvecv)
#Logitboost
library(caTools)
ctrl <- trainControl(classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
logitboostmod <- train(Made.Donation.in.March.2007~Months.since.Last.Donation+Number.of.Donations+Months.since.First.Donation,  data = traindf, method = "LogitBoost",family = "binomial", tuneLength=50,trControl = ctrl, metric = "ROC")

# summarize results
summary(logitboostmod)
#Predict on training data
lbProbs1 <- predict(logitboostmod, newdata=traindf, type="prob")[,1]
lbClasses1 <- predict(logitboostmod, newdata=traindf)
confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelb1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = lbProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelb1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LogitBoost"), fill = c("red"))
auc(rocCurvelb1)
#Predict on testing data
lbProbs <- predict(logitboostmod, newdata=testdf, type="prob")[,1]
lbClasses <- predict(logitboostmod, newdata=testdf)
confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelb <- roc (response = testdf$Made.Donation.in.March.2007, predictor = lbProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelb, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LogitBoost"), fill = c("red"))
auc(rocCurvelb)

# plot ROC curves for training 
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve1, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Training Data")
lines(rocCurvebag1, col="blue")
lines(rocCurveboost1, col="orange")
lines(rocCurverf1, col="light green")
lines(rocCurvecv1,col="yellow")
lines(rocCurvelb1,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))
# plot ROC curves for testing
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Testing Data")
lines(rocCurvebag, col="blue")
lines(rocCurveboost, col="orange")
lines(rocCurverf, col="light green")
lines(rocCurvecv,col="yellow")
lines(rocCurvelb,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))


#Storing Results in same Matrix fir Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve)[1])
m2 = cbind(confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag)[1])
m3 = cbind(confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost)[1])
m4 = cbind(confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf)[1])
m5 = cbind(confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv)[1])
m6 = cbind(confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
results <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results) <- models
names(results) <- c(stats)
results


#Storing Results in same Matrix for Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve1)[1])
m2 = cbind(confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag1)[1])
m3 = cbind(confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost1)[1])
m4 = cbind(confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf1)[1])
m5 = cbind(confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv1)[1])
m6 = cbind(confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb1)[1])
results1 <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results1) <- models
names(results1) <- c(stats)
results1


#LDA
ctrl <- trainControl(classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
ldamod <- train(Made.Donation.in.March.2007~.,data = traindf, method = "lda",family = "binomial",trControl = ctrl, metric = "ROC")
ldamod
traindf$Total.Volume.Donated..c.c..=NULL
# summarize results
summary(ldamod)
#Predict on training data
ldaProbs1 <- predict(ldamod, newdata=traindf, type="prob")[,1]
ldaClasses1 <- predict(ldamod, newdata=traindf)
confusionMatrix(data=ldaClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvelda1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = ldaProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelda1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LDA"), fill = c("red"))
auc(rocCurvelda1)
#Predict on testing data
ldaProbs <- predict(ldamod, newdata=testdf, type="prob")[,1]
ldaClasses <- predict(ldamod, newdata=testdf)
confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvelda <- roc (response = testdf$Made.Donation.in.March.2007, predictor = ldaProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvelda, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("LDA"), fill = c("red"))
auc(rocCurvelda)

#svm RBF
# Training SVM Models
library(caret)
library(dplyr)         # Used by caret
library(kernlab)       # support vector machine 
library(pROC)	       # plot the ROC curves
trainX=traindf[,1:3]
names(trainX)
# Setup for cross validation
ctrl <- trainControl(method="repeatedcv",   # 10fold cross validation
                     repeats=5,		    # do 5 repititions of cv
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE)
svcvmod <- train(x=trainX,
                 y= traindf$Made.Donation.in.March.2007,
                 method = "svmRadial",   # Radial kernel
                 tuneLength = 9,					# 9 values of the cost function
                 preProc = c("center","scale"),  # Center and scale data
                 metric="ROC",
                 trControl=ctrl)
svcvmod
#Predict on training data
svcvProbs1 <- predict(svcvmod, newdata=trainX, type="prob")[,1]
svcvClasses1 <- predict(svcvmod, newdata=trainX)
confusionMatrix(data=svcvClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvesvcv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = svcvProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesvcv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesvcv1)
#Predict on testing data
names(testdf)
testdf$Total.Volume.Donated..c.c..=NULL
testX=testdf[,1:3]
names(testX)
svcvProbs <- predict(svcvmod, newdata=testX, type="prob")[,1]
svcvClasses <- predict(svcvmod, newdata=testX)
confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvesvcv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = svcvProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesvcv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesvcv)
#SVM wihout crossvalidation
ctrl <- trainControl(summaryFunction=twoClassSummary,	classProbs=TRUE)
svmod1 <- train(x=trainX,
                y= traindf$Made.Donation.in.March.2007,
                method = "svmRadial",   # Radial kernel
                tuneLength = 9,					# 9 values of the cost function
                preProc = c("center","scale"),  # Center and scale data
                metric="ROC",
                trControl=ctrl)
svmod1

#Predict on training data
svProbs1 <- predict(svmod1, newdata=trainX, type="prob")[,1]
svClasses1 <- predict(svmod1, newdata=trainX)
confusionMatrix(data=svClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvesv1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = svProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesv1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesv1)
#Predict on testing data
svProbs <- predict(svmod1, newdata=testX, type="prob")[,1]
svClasses <- predict(svmod1, newdata=testX)
confusionMatrix(data=svClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvesv <- roc (response = testdf$Made.Donation.in.March.2007, predictor = svProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvesv, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("SVM(RBF)"), fill = c("red"))
auc(rocCurvesv)

#


# plot ROC curves for training 
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve1, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Training Data")
lines(rocCurvebag1, col="blue")
lines(rocCurveboost1, col="orange")
lines(rocCurverf1, col="light green")
lines(rocCurvecv1,col="yellow")
lines(rocCurvelb1,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))
# plot ROC curves for testing
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Testing Data")
lines(rocCurvebag, col="blue")
lines(rocCurveboost, col="orange")
lines(rocCurverf, col="light green")
lines(rocCurvecv,col="yellow")
lines(rocCurvelb,col="dark green")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.8
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost")
       , fill=c("red","blue","orange","light green","yellow","dark green"))
#ANN MLP
#install.packages("RSNNS")
library(RSNNS)
ctrl <- trainC1
control(summaryFunction=twoClassSummary,	classProbs=TRUE)
mlpmod1 <- train(Made.Donation.in.March.2007~.,
                 traindf,
                 method = "mlp",
                 preProc = c("center","scale"),  # Center and scale data
                 metric="ROC",
                 trControl=ctrl)
mlpmod1

#Predict on training data
mlpProbs1 <- predict(mlpmod1, newdata=trainX, type="prob")[,1]
mlpClasses1 <- predict(mlpmod1, newdata=trainX)
confusionMatrix(mlpClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvemlp1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = mlpProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvemlp1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("MLP"), fill = c("red"))
auc(rocCurvemlp1)
#Predict on testing data
mlpProbs <- predict(mlpmod1, newdata=testX, type="prob")[,1]
mlpClasses <- predict(mlpmod1, newdata=testX)
confusionMatrix(mlpClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvemlp <- roc (response = testdf$Made.Donation.in.March.2007, predictor = mlpProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvemlp, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("MLP"), fill = c("red"))
auc(rocCurvemlp)

#PNN


#c5.0
install.packages("C50")
library(C50)
library(caret)
traindf
names(traindf)
ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
#c5mod1 = train(Made.Donation.in.March.2007~.,trainmethod = "C5.0", preProc = c("center","scale"),metric="ROC")
c5mod1 <- train(Made.Donation.in.March.2007~.,
                traindf,
                method = "C5.0",
                preProc = c("center","scale"),  # Center and scale data
                metric="ROC",
                trControl=ctrl)

#Predict on training data
C5Probs1 <- predict(c5mod1, newdata=traindf, type="prob")[,1]
c5Classes1 <- predict(c5mod1, newdata=traindf)
confusionMatrix(c5Classes1, traindf$Made.Donation.in.March.2007)
#find the roc curve on testing data

rocCurvec51 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = C5Probs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvec51, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("C5"), fill = c("red"))
auc(rocCurvec51)
#Predict on testing data
textX=testdf[,1:3]
textX
c5Probs <- predict(c5mod1, newdata=testdf, type="prob")[,1]
c5Classes <- predict(c5mod1, newdata=testdf)
confusionMatrix(c5Classes, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvec5 <- roc (response = testdf$Made.Donation.in.March.2007, predictor = c5Probs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvec5, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("C5"), fill = c("red"))
auc(rocCurvec5)
#CART
library(caret)
library(rpart)
cartmod1 <- train(Made.Donation.in.March.2007~.,
                  traindf,
                  method = "rpart",
                  preProc = c("center","scale"),  # Center and scale data
                  metric="ROC",
                  trControl=ctrl)
summary(cartmod1)

#Predict on training data
cartProbs1 <- predict(cartmod1, newdata=traindf, type="prob")[,1]
cartClasses1 <- predict(cartmod1, newdata=traindf)
confusionMatrix(cartClasses1, traindf$Made.Donation.in.March.2007)
#find the roc curve on traininging data
library(pROC)

rocCurvecart1 <- roc (response = traindf$Made.Donation.in.March.2007, predictor = cartProbs1, levels = rev(levels(traindf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecart1, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("Cart"), fill = c("red"))
auc(rocCurvecart1)
#Predict on testing data
cartProbs <- predict(cartmod1, newdata=testdf, type="prob")[,1]
cartClasses <- predict(cartmod1, newdata=testdf)
confusionMatrix(cartClasses, testdf$Made.Donation.in.March.2007)
#find the roc curve on testing data
rocCurvecart <- roc (response = testdf$Made.Donation.in.March.2007, predictor = cartProbs, levels = rev(levels(testdf$Made.Donation.in.March.2007)))
par(mfrow=c(1,1))
plot(rocCurvecart, legacy.axes=T, col = "red", main = "Receiver Operating Characteristics (ROC) Curve")
legend("bottomright", inset=0, title="Model", border="white", bty= "n", cex = .8, legend = c("CART"), fill = c("red"))
auc(rocCurvecart)

# plot ROC curves for training 
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve1, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Training Data")
lines(rocCurvebag1, col="blue")
lines(rocCurveboost1, col="orange")
lines(rocCurverf1, col="light green")
lines(rocCurvecv1,col="yellow")
lines(rocCurvelb1,col="dark green")
lines(rocCurvelda1,col="light blue")
lines(rocCurvesvcv1,col="brown")
lines(rocCurvesv1,col="black")
lines(rocCurvemlp1,col="dark grey")
lines(rocCurvecart1,col="pink")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.5
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost", "LDA","SVM (RBF)"
                  ,"SVM","ANN MLP", "CART")
       , fill=c("red","blue","orange","light green","yellow","dark green","light blue","brown","black"
                ,"dark grey","pink"))
# plot ROC curves for testing
par(mfrow=c(1,1)) # reset graphics parameter to 1 plot
plot(rocCurve, legacy.axes=T, col="red"
     , main="Receiver Operating Characteristic (ROC) Curve for Testing Data")
lines(rocCurvebag, col="blue")
lines(rocCurveboost, col="orange")
lines(rocCurverf, col="light green")
lines(rocCurvecv,col="yellow")
lines(rocCurvelb,col="dark green")
lines(rocCurvelda,col="light blue")
lines(rocCurvesvcv,col="brown")
lines(rocCurvesv,col="black")
lines(rocCurvemlp,col="grey")
lines(rocCurvecart,col="pink")
legend("bottomright", inset=0, title="Model", border="white", bty="n", cex=.5
       , legend=c("Logit","Bagging","Boosting","Random Forest","Cross Validation","LogitBoost", "LDA","SVM (RBF)"
                  ,"SVM","ANN MLP", "CART")
       , fill=c("red","blue","orange","light green","yellow","dark green","light blue","brown","black","grey","pink"))


#Storing Results in same Matrix fir Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve)[1])
m2 = cbind(confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag)[1])
m3 = cbind(confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost)[1])
m4 = cbind(confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf)[1])
m5 = cbind(confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv)[1])
m6 = cbind(confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m7 = cbind(confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m8 = cbind(confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m9 = cbind(confusionMatrix(data=svClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m10 = cbind(confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=mlpClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
m11 = cbind(confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=cartClasses, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
results <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results) <- models
names(results) <- c(stats)
results


#Storing Results in same Matrix for Training

models = c("Logit","Bagging","Boosting","RandomForest","CrossValidation","LogitBoost")
stats = c("Accuracy","Sensitivity","Specificity","AUC")
m1 = cbind(confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=logitClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurve1)[1])
m2 = cbind(confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=bagClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvebag1)[1])
m3 = cbind(confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=boostClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurveboost1)[1])
m4 = cbind(confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=rfClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurverf1)[1])
m5 = cbind(confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=cvClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvecv1)[1])
m6 = cbind(confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=lbClasses1, traindf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb1)[1])
m7 = cbind(confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=ldaClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m8 = cbind(confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcvClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m9 = cbind(confusionMatrix(data=svClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
           ,confusionMatrix(data=svcClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
           ,confusionMatrix(data=svcClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
           ,auc(rocCurvelb)[1])
m10 = cbind(confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=mlpClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
m11 = cbind(confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$overall["Accuracy"][[1]]
            ,confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$byClass["Sensitivity"][[1]]
            ,confusionMatrix(data=cartClasses1, testdf$Made.Donation.in.March.2007)$byClass["Specificity"][[1]]
            ,auc(rocCurvelb)[1])
results1 <- data.frame(rbind(m1,m2,m3,m4,m5,m6))
row.names(results1) <- models
names(results1) <- c(stats)
results1
