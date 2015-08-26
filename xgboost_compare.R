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
dummies <- dummyVars(~ ., data = X_train, fullRank=TRUE)
X_train_exp <- as.matrix(predict(dummies, newdata = X_train))
storage.mode(X_train_exp) <- "double"
trainData <- xgb.DMatrix(X_train_exp, label = y_train)
#commandArgs
#commandArgs
test_data <- read.csv('data/test.csv')
X_test<-subset(test_data,select=-c(Id))
X_test_exp <- as.matrix(predict(dummies, newdata = X_test))
storage.mode(X_train_exp) <- "double"
testData <- xgb.DMatrix(X_test_exp)

evalerror <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    err <- NormalizedGini(labels, preds)
    return(list(metric = "xxxx", value = err))
}

trainModels <-function(seed) {
    set.seed(seed)
    folds <- createFolds(train_data$Hazard,k=nfold, list=TRUE, returnTrain=FALSE)
    taskParams <- data.frame(max.depth=9,eta=0.005, subsample=0.7,colsample_bytree=0.7,min_child_weight=6,gamma=0)
    print("Running task")
    print(taskParams)

    techParams <- list(objective = "reg:linear")

    params <- c(taskParams, techParams)
    cv_gini <-xgb.cv(params = params, data = trainData,nrounds = nrounds, folds = folds, prediction = TRUE, 
            early.stop.round = 120,  maximize = TRUE, nthread=16, eval_metric=evalerror)
    min_nrounds_gini <- which.max(as.numeric(cv_gini$dt$test.xxxx.mean))
    gini_gini <-sapply(folds,function(f){NormalizedGini(y_train[f], cv_gini$pred[f])})

    cv_rmse <-xgb.cv(params = params, data = trainData,nrounds = nrounds, folds = folds, prediction = TRUE, 
            early.stop.round = 120,  maximize = FALSE, nthread=16, eval_metric='rmse')
    min_nrounds_rmse <- which.min(as.numeric(cv_rmse$dt$test.rmse.mean))
    gini_rmse <-sapply(folds,function(f){NormalizedGini(y_train[f], cv_rmse$pred[f])})
    
    res <- data.frame(gini.round = min_nrounds_gini,
               gini.fold.mean = mean(gini_gini), 
               gini.fold.sd = sd(gini_gini),
               gini.total = NormalizedGini(y_train, cv_gini$pred),
               rmse.round = min_nrounds_rmse,
               rmse.fold.mean = mean(gini_rmse),
               rmse.fold.sd = sd(gini_rmse),
               rmse.total = NormalizedGini(y_train, cv_rmse$pred)
    )
    print(res)
    res
}

set.seed(37)
seeds <- sample(100,10,replace=FALSE)
result <- do.call('rbind',lapply(seeds, trainModels))
print(result)
summary(result)

