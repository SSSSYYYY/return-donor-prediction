### load the original data
testB = read.table(file="https://raw.githubusercontent.com/Hongxia555/Blood-Donation/master/train.csv", header=TRUE, sep=",")
compareTable = read.table(file="https://raw.githubusercontent.com/Hongxia555/Blood-Donation/master/testtable.csv", header=TRUE, sep="\t")
clusterTable = read.table(file="https://raw.githubusercontent.com/Hongxia555/Blood-Donation/master/cluster_3.csv", header=TRUE, sep=',')
colnames(clusterTable)[1] <- "Type"
clusterTable[,1] <- c("Logistic Regression", "Logit(Bagged)", "Logit(Boosted)", "RandomForest", "Logit(10-fold CV)",
                            "LogitBoost", "LDA", "SVM(5-fold CV)", "SVM", "C5.0", "CART", "MLP")
library(shiny)
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
library(ggplot2)
library(rpart)
library(RCurl)
library(dplyr)
library(wesanderson)
library(dplyr)
library(radarchart)
options(rgl.printRglwidget = TRUE)

#Dropping the ID Variable
testB$X=NULL
#Drop the volume donated
testB$Total.Volume.Donated..c.c..=NULL

# split data set into training and testing
set.seed(2016)
split <- createDataPartition(testB$Made.Donation.in.March.2007, p=0.7, list=F)
traindf <- testB[split,]
testdf <- testB[-split,]

#Set function for weighted average cost/benefit

weightedAve <- function(Opos,Oneg,Apos,Aneg,Bpos,Bneg,ABpos,ABneg){
  0.355*Opos + 0.335*Apos + 0.095*Bpos + 0.075*Oneg + 0.06*Aneg + 0.05*ABpos + 0.015*Bneg + 0.015*ABneg
}


### prediction models

################################# logistic ##############################

ctrl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)

logit <- train(Made.Donation.in.March.2007~.,
               data = traindf,
               method = "glm",
               family = "binomial",
               metric = "ROC",
               trControl = ctrl)

#Finding confusion Matrix for Testing Data

logitProbs <- predict(logit, newdata=testdf, type="prob")[,1]
logitClasses <- predict(logit, newdata=testdf)
confusionMatrix(data=logitClasses, testdf$Made.Donation.in.March.2007)



########################## logistic boosted  #############################

boostmod=train(Made.Donation.in.March.2007~.,
               data=traindf,
               method="gbm",
               verbose=F,
               trControl=trainControl(method="cv",number=60))

#Creating confusion matrix for testing data

boostProbs <- predict(boostmod, newdata=testdf, type="prob")[,1]
boostClasses <- predict(boostmod, newdata=testdf)
confusionMatrix(data=boostClasses, testdf$Made.Donation.in.March.2007)



##########################  logistic bagged #########################

bagmod=train(Made.Donation.in.March.2007~.,
             data=traindf,
             method="treebag",
             trControl=trainControl(method="cv",number=35))

#inding consusion matrix on testing data

bagProbs <- predict(bagmod, newdata=testdf, type="prob")[,1]
bagClasses <- predict(bagmod, newdata=testdf)
confusionMatrix(data=bagClasses, testdf$Made.Donation.in.March.2007)


############################### Logitboost ################################

library(caTools)

# train the model
logitboostmod <- train(Made.Donation.in.March.2007~.,
                       data = traindf,
                       method = "LogitBoost",
                       family = "binomial",
                       tuneLength=50,
                       trControl = ctrl,
                       metric = "ROC")

#Predict on testing data

lbProbs <- predict(logitboostmod, newdata=testdf, type="prob")[,1]
lbClasses <- predict(logitboostmod, newdata=testdf)
confusionMatrix(data=lbClasses, testdf$Made.Donation.in.March.2007)


###########################  logistic 10-fold cv ##########################
# 10-fold CV

ctrl10 <- trainControl(method = "cv", number=10,classProbs =TRUE, summaryFunction = twoClassSummary)

# train the model
CVmod <- train(Made.Donation.in.March.2007~.,
               data = traindf,
               method = "nb",
               tuneLength =10,
               family = "binomial",
               trControl = ctrl10
               , metric = "ROC")


#Predict on testing data
cvProbs <- predict(CVmod, newdata=testdf, type="prob")[,1]
cvClasses <- predict(CVmod, newdata=testdf)
confusionMatrix(data=cvClasses, testdf$Made.Donation.in.March.2007)


########################### random forest  ############################

RFmod=train(Made.Donation.in.March.2007~.,
            data=traindf,
            method="rf",
            importance=T,
            trControl=trainControl(method="cv",number=50))

#finding confusion matrix for testing data
rfProbs <- predict(RFmod, newdata=testdf, type="prob")[,1]
rfClasses <- predict(RFmod, newdata=testdf)
confusionMatrix(data=rfClasses, testdf$Made.Donation.in.March.2007)


############################## LDA ################################

ctrl <- trainControl(classProbs =TRUE, summaryFunction = twoClassSummary)
# train the model
ldamod <- train(Made.Donation.in.March.2007~.,data = traindf, method = "lda",family = "binomial",trControl = ctrl, metric = "ROC")
ldamod
traindf$Total.Volume.Donated..c.c..=NULL

#Predict on testing data
ldaProbs <- predict(ldamod, newdata=testdf, type="prob")[,1]
ldaClasses <- predict(ldamod, newdata=testdf)
confusionMatrix(data=ldaClasses, testdf$Made.Donation.in.March.2007)


############################   SVM RBF #######################
# Training SVM Models
library(kernlab)       # support vector machine

trainX=traindf[,1:3]

# Setup for cross validation
ctrlSVM <- trainControl(method="repeatedcv",   # 10fold cross validation
                        repeats=5,		    # do 5 repititions of cv
                        summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                        classProbs=TRUE)

svcvmod <- train(x=trainX,
                 y= traindf$Made.Donation.in.March.2007,
                 method = "svmRadial",   # Radial kernel
                 tuneLength = 9,					# 9 values of the cost function
                 preProc = c("center","scale"),  # Center and scale data
                 metric="ROC",
                 trControl=ctrlSVM)

#Predict on testing data
testX=testdf[,1:3]
svcvProbs <- predict(svcvmod, newdata=testX, type="prob")[,1]
svcvClasses <- predict(svcvmod, newdata=testX)
confusionMatrix(data=svcvClasses, testdf$Made.Donation.in.March.2007)

# #SVM wihout crossvalidation

svmod1 <- train(x=trainX,
                y= traindf$Made.Donation.in.March.2007,
                method = "svmRadial",   # Radial kernel
                tuneLength = 9,					# 9 values of the cost function
                preProc = c("center","scale"),  # Center and scale data
                metric="ROC",
                trControl=ctrl)

#Predict on testing data

svProbsmod1 <- predict(svmod1, newdata=testX, type="prob")[,1]
svClasses <- predict(svmod1, newdata=testX)
confusionMatrix(data=svClasses, testdf$Made.Donation.in.March.2007)


# ##########################   CART      ########################
cartmod1 <- train(Made.Donation.in.March.2007~.,
                  traindf,
                  method = "rpart",
                  preProc = c("center","scale"),  # Center and scale data
                  metric="ROC",
                  trControl=ctrl)

#Predict on testing data

cartProbs <- predict(cartmod1, newdata=testX, type="prob")[,1]
cartClasses <- predict(cartmod1, newdata=testX)
confusionMatrix(cartClasses, testdf$Made.Donation.in.March.2007)




### Define server logic required to draw a histogram
shinyServer(function(input, output) {
  
  ##Risk preference - score matrix
  #tableOutput
  output$CnBMatrix <- renderTable({
    
    #weighted average Cost/Benefit of TP, FN, FP, TN using test data
    
    TP <- weightedAve(input$OposTP,input$OnegTP,input$AposTP,input$AnegTP,input$BposTP,input$BnegTP,input$ABposTP,input$ABnegTP)
    FN <- weightedAve(input$OposFN,input$OnegFN,input$AposFN,input$AnegFN,input$BposFN,input$BnegFN,input$ABposFN,input$ABnegFN)
    FP <- weightedAve(input$OposFP,input$OnegFP,input$AposFP,input$AnegFP,input$BposFP,input$BnegFP,input$ABposFP,input$ABnegFP)
    TN <- weightedAve(input$OposTN,input$OnegTN,input$AposTN,input$AnegTN,input$BposTN,input$BnegTN,input$ABposTN,input$ABnegTN)
    
    matrixTable = data.frame(matrix(1:4, nrow=2, ncol=2))
    
    matrixTable[1,1] <- TP
    matrixTable[1,2] <- FP
    matrixTable[2,1] <- FN
    matrixTable[2,2] <- TN
    
    colnames(matrixTable) <- c("Actual Positive", "Actual Negative")
    rownames(matrixTable) <- c("Predicted Positive", "Predicted Negative")
    
    matrixTable 
  },
  include.rownames = TRUE)
  
  # ##Results - benefit plot
  # #plotOutput
  output$benefitgraph <- renderPlot({

    TP <- weightedAve(input$OposTP,input$OnegTP,input$AposTP,input$AnegTP,input$BposTP,input$BnegTP,input$ABposTP,input$ABnegTP)
    FN <- weightedAve(input$OposFN,input$OnegFN,input$AposFN,input$AnegFN,input$BposFN,input$BnegFN,input$ABposFN,input$ABnegFN)
    FP <- weightedAve(input$OposFP,input$OnegFP,input$AposFP,input$AnegFP,input$BposFP,input$BnegFP,input$ABposFP,input$ABnegFP)
    TN <- weightedAve(input$OposTN,input$OnegTN,input$AposTN,input$AnegTN,input$BposTN,input$BnegTN,input$ABposTN,input$ABnegTN)

    ##all models

    ###################LOGIT##################
    x <- 0.1
    logit_finalresults <- NULL
    while( x < 0.8){
      x = x + 0.1
      logit_result_Approved <- ifelse(logitProbs >= x , 1, 0)
      cm_logit <- table(logit_result_Approved,testdf$Made.Donation.in.March.2007, dnn = c("predict", "actual"))
      benefit<- (cm_logit[1,1]*TN + cm_logit[1,2]*FN + cm_logit[2,1]*FP + cm_logit[2,2]*TP)
      type <- "Logistic Regression"
      logit_finalresults <-rbind(logit_finalresults,data.frame(x,benefit,type))
    }

    ###################LOGIT BOOST####################
    x <- 0.1
    logitB_finalresults <- NULL
    while( x < 0.8){
        x = x + 0.1
        logitB_result_Approved <- ifelse(boostProbs >= x, 1, 0)
        cm_logitB <- table(logitB_result_Approved,testdf$Made.Donation.in.March.2007, dnn = c("predict", "actual"))
        benefit <- (cm_logitB[1,1]*TN + cm_logitB[1,2]*FN + cm_logitB[2,1]*FP + cm_logitB[2,2]*TP)
        type <- "Logit(Boosted)"
        logitB_finalresults <-rbind(logitB_finalresults,data.frame(x,benefit,type))
    }


    ##########################  logistic bagged #########################
    x <- 0.1
    logitBG_finalresults <- NULL
    while( x < 0.8){
      x = x + 0.1
      logitBG_result_Approved <- ifelse(bagProbs >= x , 1, 0)
      logitBG_cm <- table(logitBG_result_Approved,testdf$Made.Donation.in.March.2007, dnn = c("predict", "actual"))
      benefit<- (logitBG_cm[1,1]*TN + logitBG_cm[1,2]*FN + logitBG_cm[2,1]*FP + logitBG_cm[2,2]*TP)
      type <- "Logit(Bagged)"
      logitBG_finalresults <-rbind(logitBG_finalresults,data.frame(x,benefit,type))
    }


    ############################### Logitboost ################################
    x <- 0.1
    logitBT_finalresults <- NULL
    while( x < 0.8){
      x = x + 0.1
      logitBT_result_Approved <- ifelse(lbProbs >= x , 1, 0)
      logitBT_cm <- table(logitBT_result_Approved,testdf$Made.Donation.in.March.2007, dnn = c("predict", "actual"))
      benefit<- (logitBT_cm[1,1]*TN + logitBT_cm[1,2]*FN + logitBT_cm[2,1]*FP + logitBT_cm[2,2]*TP)
      type <- "LogitBoost"
      logitBT_finalresults <-rbind(logitBT_finalresults,data.frame(x,benefit,type))
    }


    ###########################  logistic 10-fold cv ##########################
    x <- 0.1
    logitCV_finalresults <- NULL
    while( x < 0.8){
      x = x + 0.1
      logitCV_result_Approved <- ifelse(cvProbs >= x , 1, 0)
      logitCV_cm <- table(logitCV_result_Approved,testdf$Made.Donation.in.March.2007, dnn = c("predict", "actual"))
      benefit<- (logitCV_cm[1,1]*TN + logitCV_cm[1,2]*FN + logitCV_cm[2,1]*FP + logitCV_cm[2,2]*TP)
      type <- "Logit(10-fold CV)"
      logitCV_finalresults <-rbind(logitCV_finalresults,data.frame(x,benefit,type))
    }


    ########################## random forest  ############################
    x <- 0.1
    RF_finalresults <- NULL
    while( x < 0.8){
      x = x + 0.1
      RF_result_Approved <- ifelse(rfProbs >= x , 1, 0)
      RF_cm <- table(RF_result_Approved,testdf$Made.Donation.in.March.2007, dnn = c("predict", "actual"))
      benefit<- (RF_cm[1,1]*TN + RF_cm[1,2]*FN + RF_cm[2,1]*FP +RF_cm[2,2]*TP)
      type <- "RandomForest"
      RF_finalresults <-rbind(RF_finalresults,data.frame(x,benefit,type))
    }


    ############################## LDA ################################
    x <- 0.1
    LDA_finalresults <- NULL
    while( x < 0.8){
      x = x + 0.1
      LDA_result_Approved <- ifelse(ldaProbs >= x , 1, 0)
      LDA_cm <- table(LDA_result_Approved,testdf$Made.Donation.in.March.2007, dnn = c("predict", "actual"))
      benefit<- (LDA_cm[1,1]*TN + LDA_cm[1,2]*FN + LDA_cm[2,1]*FP +LDA_cm[2,2]*TP)
      type <- "LDA"
      LDA_finalresults <-rbind(LDA_finalresults,data.frame(x,benefit,type))
    }



    ############################   SVM RBF #######################
    x <- 0.1
    SVCV_finalresults <- NULL
    while( x < 0.8){
      x = x + 0.1
      SVCV_result_Approved <- ifelse(svcvProbs >= x , 1, 0)
      SVCV_cm <- table(SVCV_result_Approved,testdf$Made.Donation.in.March.2007, dnn = c("predict", "actual"))
      benefit<- (SVCV_cm[1,1]*TN + SVCV_cm[1,2]*FN + SVCV_cm[2,1]*FP +SVCV_cm[2,2]*TP)
      type <- "SVM(5-fold CV)"
      SVCV_finalresults <-rbind(SVCV_finalresults,data.frame(x,benefit,type))
    }


    #SVM wihout crossvalidation
    x <- 0.1
    SVPROBS_finalresults <- NULL
    while( x < 0.7){
      x = x + 0.1
      SVPROBS_result_Approved <- ifelse(svProbsmod1 >= x , 1, 0)
      SVPROBS_cm <- table(SVPROBS_result_Approved,testdf$Made.Donation.in.March.2007, dnn = c("predict", "actual"))
      benefit<- (SVPROBS_cm[1,1]*TN + SVPROBS_cm[1,2]*FN + SVPROBS_cm[2,1]*FP +SVPROBS_cm[2,2]*TP)
      type <- "SVM"
      SVPROBS_finalresults <-rbind(SVPROBS_finalresults,data.frame(x,benefit,type))
    }

    ############################   CART #######################
    x <- 0.1
    CART_finalresults <- NULL
    while( x < 0.7){
      x = x + 0.1
      CART_result_Approved <- ifelse(cartProbs >= x , 1, 0)
      CART_cm <- table(CART_result_Approved,testdf$Made.Donation.in.March.2007, dnn = c("predict", "actual"))
      benefit<- (CART_cm[1,1]*TN + CART_cm[1,2]*FN + CART_cm[2,1]*FP +CART_cm[2,2]*TP)
      type <- "CART"
      CART_finalresults <-rbind(CART_finalresults,data.frame(x,benefit,type))
    }


    #Summarize all the models
    graphData <- data.frame(rbind(logit_finalresults[1:7,],logitB_finalresults[1:7,],logitBG_finalresults[1:7,],
                                  logitBT_finalresults[1:7,],logitCV_finalresults[1:7,],RF_finalresults[1:7,],
                                  LDA_finalresults[1:7,],SVCV_finalresults[1:7,],SVPROBS_finalresults[1:7,],
                                  CART_finalresults[1:7,]))



    #Filter as input by user
    dat_reac <- reactive({
      filter(graphData, type %in% input$models)
    })


    #plot
    ggplot(dat_reac(), aes(x=x, y=benefit, color = type)) +
      geom_line(size=2) +
      xlab("cutoff")




  })
  
   
  # ##Tableoutput - Comparison
     output$stats <- renderTable({
       
       ##cluster or not, build reactive data for table
       if (input$cluster == TRUE){
         
         dat_reac <- reactive({
           filter(clusterTable, Type %in% input$models)
         })
         
       } else {
         
         dat_reac <- reactive({
           filter(compareTable, Type %in% input$models)
         })

       } 
       
       dat_reac()
       
      
   })
     
})
    

