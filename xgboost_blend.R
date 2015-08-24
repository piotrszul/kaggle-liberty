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
model <- read.table('target/output1.txt', header = FALSE, sep = ' ', as.is=TRUE)
input <-as.data.frame(t(model))
ref <- train_data$Hazard
metrics <- sapply(input, function(x){NormalizedGini(ref, x) })
desc <- order(metrics, decreasing = TRUE)
model_num <- 5
selection <- desc[1:model_num]
weights <- rep(0, length.out=length(metrics))
weights[selection] <- 1
sums <- apply(input[,selection], 1, sum)
pred <- sums/model_num
NormalizedGini(ref, pred)

for (i in 1:500 ) {
    model_num <- model_num + 1
    metrics <- sapply(input, function(x){NormalizedGini(ref, (x + sums)/model_num) })
    sel <- which.max(metrics)
    weights[sel] <- weights[sel] + 1
    sums <- sums + input[,sel]
    pred <- sums/model_num
    print(NormalizedGini(ref, pred))
}




