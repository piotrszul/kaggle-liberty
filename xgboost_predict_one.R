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

train_data <- read.csv('data/train.csv')
test_data <- read.csv('data/test.csv')
set.seed(100)
trainSet <- subset(train_data,select=-c(Id))
y_train<-trainSet$Hazard
X_train<-trainSet[,-1]
dummies <- dummyVars(~ ., data = X_train, fullRank=TRUE)
X_train_exp <- as.matrix(predict(dummies, newdata = X_train))
storage.mode(X_train_exp) <- "double"
trainData <- xgb.DMatrix(X_train_exp, label = y_train)

nrounds <- 1273 
taskParams <- data.frame(max.depth=10, eta=0.005, subsample=0.8, colsample_bytree=0.65, min_child_weight=40, gamma=1) 

print("Running task")
techParams <- list(
            objective = "reg:linear")

params <- c(taskParams, techParams)
model <-xgb.train(params = params, data = trainData,nrounds = nrounds, nthreads=16, verbose=1)

testSet <- subset(test_data,select=-c(Id))
X_test<-testSet
X_test_exp <- as.matrix(predict(dummies, newdata = X_test))
storage.mode(X_test_exp) <- "double"
pred <- predict(model, X_test_exp)
result_one <- data.frame(Id=test_data$Id, Hazard=pred)
write.csv(result_one, file.path('output','submission-one.csv'), row.names=FALSE, quote=FALSE)


