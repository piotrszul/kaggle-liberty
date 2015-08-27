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


ensambleParams <- read.csv(file.path('target','ensamble-rmse.csv'))
selectedModelParams = ensambleParams[ensambleParams$pruned_weights>0,]
print("Models selected:")
print(nrow(selectedModelParams))

thisModelParams <- selectedModelParams[taskId,]
nrounds <- thisModelParams$nrounds
taskParams <-thisModelParams[c('max.depth', 'eta', 'subsample', 'colsample_bytree', 'min_child_weight','gamma')]

taskName <-paste(names(taskParams), taskParams, collapse='+', sep='_')

print("Running task")
print(taskId)
print(taskName)
print(thisModelParams)

techParams <- list(
            objective = "reg:linear"
            )

params <- c(taskParams, techParams)
model <-xgb.train(params = params, data = trainData,nrounds = nrounds, nthreads=16)

testSet <- subset(test_data,select=-c(Id))
X_test<-testSet
X_test_exp <- as.matrix(predict(dummies, newdata = X_test))
storage.mode(X_test_exp) <- "double"
pred <- predict(model, X_test_exp)
test_pred <-data.frame(pred=pred)
save(test_pred, file=paste('target/task/test-', taskName, sep=''), compress='gzip')




