library(xgboost)
library(caret)

multiLogLoss <-function (data,label) {
    print("multiLogLoss")
    sumProbs <- apply(data,1, sum)
    classProb <- apply(cbind(as.numeric(label), data),1,function(x) {x[x[1]+1]})
    scaledProb <- classProb/sumProbs
    adjustedProb <- pmax(pmin(scaledProb,1-(1e-15)),1e-15)
    logLoss = - mean(log(adjustedProb))
    print(logLoss)
    c(LOGLOSS=logLoss)
}

#"NormalizedGini" is the other half of the metric. This function does most of the work, though
SumModelGini <- function(solution, submission) {
    df = data.frame(solution = solution, submission = submission)
    df <- df[order(df$submission, decreasing = TRUE),]
    df
    df$random = (1:nrow(df))/nrow(df)
    df
    totalPos <- sum(df$solution)
    df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
    df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
    df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
    #print(df)
    return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
    SumModelGini(solution, submission) / SumModelGini(solution, solution)
}


taskId <- as.numeric(commandArgs(trailingOnly=TRUE)[1])
if (is.na(taskId)) {
    taskId <- 1
}

nrounds <- 15000
nfold <- 10 
nthread <- 16

train_data <- read.csv('data/train.csv')
set.seed(100)
cov_and_res  = subset(train_data,select=-c(Id))
trainSet <- cov_and_res
y_train<-trainSet$Hazard
X_train<-trainSet[,-1]
dummies <- dummyVars(~ ., data = X_train, fullRank=TRUE)
X_train_exp <- as.matrix(predict(dummies, newdata = X_train))
storage.mode(X_train_exp) <- "double"
trainData <- xgb.DMatrix(X_train_exp, label = y_train)
#commandArgs

#1.paramGrid <- expand.grid(max.depth=c(8, 10, 12), eta=c(0.01,0.008), subsample=c(1,0.75,0.5), colsample_bytree=c(1,0.75,0.5))
#2.paramGrid <- expand.grid(max.depth=c(1,5,15,30), eta=c(0.1,0.001,0.0001), subsample=c(1, 0.75), colsample_bytree=c(1,0.75,0.5), min_child_weight=c(1,5,10))
#3.paramGrid <- expand.grid(max.depth=c(5,15,30), eta=c(0.001,0.0001), subsample=c(1, 0.75, 0.5), colsample_bytree=c(1,0.75,0.5), min_child_weight=c(1,5,10))
#4.paramGrid <- expand.grid(max.depth=c(3,5,10), eta=c(0.005,0.001,0.0005), subsample=c(1, 0.75), colsample_bytree=c(1,0.75), min_child_weight=c(1,10,20))
#5.paramGrid <- expand.grid(max.depth=c(5,6,7,8,9,10,11,12), eta=c(0.001), subsample=c(0.85, 0.75, 0.65), colsample_bytree=c(1,0.75, 0.6), min_child_weight=c(15,20,25))
#6.paramGrid <- expand.grid(max.depth=c(8,9,10,11), eta=c(0.001), subsample=c(0.8, 0.75,0.7), colsample_bytree=c(0.75, 0.6,0.5), min_child_weight=c(25,30), gamma=c(0,1))
#7.paramGrid <- expand.grid(max.depth=c(8,9,10), eta=c(0.001), subsample=c(0.8, 0.75), colsample_bytree=c(0.65, 0.6,0.55), min_child_weight=c(30,35,40), gamma=c(1,5,10))
paramGrid <- expand.grid(max.depth=c(8,9,10), eta=c(0.001), subsample=c(0.8, 0.75), colsample_bytree=c(0.65, 0.6,0.55), min_child_weight=c(30,35,40), gamma=c(1,5,10))

str(paramGrid)
taskParams <- as.list(paramGrid[taskId,])

evalerror <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    err <- NormalizedGini(labels, preds)
    return(list(metric = "xxxx", value = err))
}


print("Running task")
print(taskId)
print(taskParams)


techParams <- list(
#	     nthread= nthread,
            objective = "reg:linear", 
#            early.stop.round = 10,
#            maximize = TRUE,
            eval_metric = evalerror)

params <- c(taskParams, techParams)
cv <-xgb.cv(params = params, data = trainData,nrounds = nrounds, nfold = nfold, early.stop.round = 200,  maximize = TRUE, nthread=16)
min_nrounds <- which.max(as.numeric(cv$test.xxxx.mean))
result <- data.frame(rmse = as.numeric(cv$test.xxxx.mean[min_nrounds]), 
                    min_nrounds = min_nrounds, taskParams, id=taskId)         
print(result)
write.table(result, paste('target/task/out_', taskId, sep=''), row.names=FALSE, quote=FALSE, col.names=FALSE)




