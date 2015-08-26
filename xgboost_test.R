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
nfold <- 5 
nthread <- 16

train_data <- read.csv('data/train.csv')
set.seed(100)
cov_and_res  = subset(train_data,select=-c(Id))
trainIdx <-createDataPartition(train_data$Hazard, p=0.75, list=FALSE)
trainSet <- cov_and_res[trainIdx, ]
y_train<-trainSet$Hazard
X_train<-trainSet[,-1]
set.seed(100)
folds <- createFolds(y_train,k=nfold, list=TRUE, returnTrain=FALSE)
dummies <- dummyVars(~ ., data = X_train, fullRank=TRUE)
X_train_exp <- as.matrix(predict(dummies, newdata = X_train))
storage.mode(X_train_exp) <- "double"
trainData <- xgb.DMatrix(X_train_exp, label = y_train)
#commandArgs
testSet <- cov_and_res[-trainIdx, ]
y_test<-testSet$Hazard
X_test<-testSet[,-1]
X_test_exp <- as.matrix(predict(dummies, newdata = X_test))
storage.mode(X_train_exp) <- "double"
testData <- xgb.DMatrix(X_test_exp, label = y_test)

taskParams <- data.frame(max.depth=5,eta=0.10, subsample=0.66,colsample_bytree=1.00,min_child_weight=60,gamma=10)
taskName <-paste(names(taskParams), taskParams, collapse='+', sep='_')
evalerror <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    err <- NormalizedGini(labels, preds)
    #err <- 1/RMSE(preds,labels)
    return(list(metric = "xxxx", value = err))
}

print("Running task")
print(taskId)
print(taskParams)
print(taskName)

techParams <- list(
            objective = "reg:linear", 
            eval_metric = evalerror)

params <- c(taskParams, techParams)
set.seed(100)
cv <-xgb.cv(params = params, data = trainData,nrounds = nrounds, folds = folds, prediction = TRUE, early.stop.round = 50,  maximize = TRUE, nthread=16)
min_nrounds <- which.max(as.numeric(cv$dt$test.xxxx.mean))
result <- data.frame(cv$dt[min_nrounds], 
                    min_nrounds = min_nrounds, taskParams, id=taskId)         
print(result)
#write.table(result, paste('target/task/out-', taskName, sep=''), row.names=FALSE, quote=FALSE, col.names=FALSE)
#save(cv, file=paste('target/task/pred-', taskName, sep=''), compress='gzip')
gini <-sapply(folds,function(f){NormalizedGini(y_train[f], cv$pred[f])})
mean(gini)
sd(gini)
set.seed(100)
model <-xgb.train(params = params, data = trainData,nrounds = nrounds, early.stop.round = 50,  maximize = TRUE, 
                  watchlist=list(test=testData, train=trainData), verbose=1)
test_pred <-predict(model, newdata=testData)
test_gini <-NormalizedGini(y_test, test_pred)
test_gini









