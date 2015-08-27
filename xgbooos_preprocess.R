library(xgboost)
library(caret)

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


nrounds <- 15000
nfold <- 2 
nthread <- 16

train_data <- read.csv('data/train.csv')
set.seed(100)

cov_and_res  = subset(train_data,select=-c(Id))
trainSet <- cov_and_res
y_train<-trainSet$Hazard
X_train<-trainSet[,-1]
#dummies <- dummyVars(~ ., data = X_train, fullRank=TRUE)
#X_train_exp <- as.matrix(predict(dummies, newdata = X_train))
#storage.mode(X_train_exp) <- "double"
#trainData <- xgb.DMatrix(X_train_exp, label = y_train)
#commandArgs
#commandArgs
test_data <- read.csv('data/test.csv')
X_test<-subset(test_data,select=-c(Id))
#X_test_exp <- as.matrix(predict(dummies, newdata = X_test))
#storage.mode(X_train_exp) <- "double"
#testData <- xgb.DMatrix(X_test_exp)


expandVar <- function(var) {
    l <- levels(var)
    if (is.null(l)) var else 
        if (length(l) >2 ) sapply(var, function(c) {as.numeric(charToRaw(as.character(c)))-65})  
    else as.numeric(var)-1
}

X_train_ex1 <- as.data.frame(lapply(X_train,expandVar))
nzv <- nearZeroVar(X_train_ex1, saveMetrics= TRUE)
nzv[nzv$nzv,][1:10,]
descrCor <-  cor(X_train_ex1)
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .9)

highlyCorDescr <- findCorrelation(X_train_ex1, cutoff = .75)
filteredDescr <- X_train_ex1[,-highlyCorDescr]
descrCor2 <- cor(X_train_ex1)
summary(descrCor2[upper.tri(descrCor2)])


