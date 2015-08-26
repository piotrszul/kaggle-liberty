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
y_train<-train_data$Hazard
set.seed(100)
folds <- createFolds(train_data$Hazard,k=5, list=TRUE, returnTrain=FALSE)
test_data <- read.csv('data/test.csv')
ensambleParams <- read.csv(file.path('target','ensamble.csv'),header=TRUE)
selectedModels <- ensambleParams$pruned_weights>0
selectedModelParams = ensambleParams[selectedModels,]
print("Models selected:")
print(nrow(selectedModelParams))
modelIds<-selectedModelParams[,1]
inputList <-lapply(modelIds, function(f) { load(file.path('target', 'task', paste('test',f, sep='-'))); test_pred$pred })
input <- data.frame(inputList)
tinputList <-lapply(modelIds, function(f) { load(file.path('target', 'task', paste('pred',f, sep='-'))); cv$pred })
tinput <- data.frame(tinputList)
weights <- selectedModelParams$pruned_weights
normWeights <-weights/sum(weights)
pred <-apply(input, MARGIN=1, FUN=function(x){ sum(normWeights*x)})
result <- data.frame(Id=test_data$Id, Hazard=pred)
write.csv(result, file.path('output','submission.csv'), row.names=FALSE, quote=FALSE)
max_model<-5
selectedModelParams[max_model,]
result_max <- data.frame(Id=test_data$Id, Hazard=input[,max_model])
write.csv(result_max, file.path('output','submission-max.csv'), row.names=FALSE, quote=FALSE)
NormalizedGini(y_train, tinput[,max_model])
sapply(folds, function(f){NormalizedGini(y_train[f], tinput[f,max_model])})
sapply(folds, function(f){RMSE(y_train[f], tinput[f,max_model])})




