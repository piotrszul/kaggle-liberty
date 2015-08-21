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

nrounds <- 7996
nfold <- 5
nthread <-3

train_data <- read.csv('data/train.csv')
test_data <- read.csv('data/test.csv')
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
taskParams <- data.frame(max.depth=9, eta=0.001, subsample=0.75, colsample_bytree=0.55, min_child_weight=40, gamma=5)

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
            eval_metric = evalerror)

params <- c(taskParams, techParams)
model <-xgb.train(params = params, data = trainData,nrounds = nrounds, nthreads=4)

testSet <- subset(test_data,select=-c(Id))
X_test<-testSet
X_test_exp <- as.matrix(predict(dummies, newdata = X_test))
storage.mode(X_test_exp) <- "double"
pred <- predict(model, X_test_exp)
#resultTest <- data.frame(bidder_id=test_data$bidder_id,prediction=pred)
result <- data.frame(Id=test_data$Id, Hazard=pred)
write.csv(result, file.path('output','submission.csv'), row.names=FALSE, quote=FALSE)




