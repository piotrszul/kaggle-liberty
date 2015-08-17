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

nrounds <- 500
nfold <- 5
nthread <-3

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

paramGrid <- expand.grid(max.depth=c(13,15,17), eta=c(0.01), subsample=c(0.75), colsample_bytree=c(0.5), min_child_weight=c(5,6,7))
#paramGrid <- expand.grid(max.depth=c(8, 10, 12), eta=c(0.01,0.008), subsample=c(1,0.75,0.5), colsample_bytree=c(1,0.75,0.5))
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


techParams <- list( nthread=4,
            objective = "reg:linear", 
            early.stop.round = 10,
            maximize = TRUE,
            eval_metric = evalerror)

params <- c(taskParams, techParams)
cv <-xgb.cv(params = params, data = trainData,nrounds = nrounds, nfold = nfold)
min_nrounds <- which.min(as.numeric(cv$test.rmse.mean))
result <- data.frame(rmse = as.numeric(cv$test.rmse.mean[min_nrounds]), 
                    min_nrounds = min_nrounds, taskParams, id=taskId)         
print(result)
write.table(result, paste('target/task/out_', taskId, sep=''), row.names=FALSE, quote=FALSE, col.names=FALSE)




