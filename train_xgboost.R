library(caret)
library(xgboost)

printf <- function(...) invisible(print(sprintf(...)))
commit <- function() {
    modelId <-paste("xgb", format(model$timestamp, "%y%m%d_%H%M%S"), sprintf('%0.5f', estAUC), sep='-' )
    printf("Committing model: %s", modelId)
    write.csv(resultTest,sprintf('models/%s.csv', modelId), quote=FALSE,row.names=FALSE)
    #write.csv(resultTrain,sprintf('models/meta/train_%s.csv', modelId), quote=FALSE,row.names=FALSE)    
    #save(model,file=sprintf('models/meta/%s.Rdata', modelId)) 
    system("git add .")
    system(sprintf('git commit -m "MODEL: %s"', modelId))
}

train_data<- read.csv('data/train.csv', header=TRUE)
test_data <- read.csv('data/test.csv', header=TRUE)
train_set <- subset(train_data,select=-c(Id, Hazard))

trainMatrix <- as.matrix(train_set)
storage.mode(trainMatrix) <- "double"
trainData <- xgb.DMatrix(trainMatrix, label = as.numeric(test_data$Hazard))

set.seed(137)
params <- list(max.depth=17,  nthread=4,
               eta = 0.0007, objective = "reg:linear", subsample=0.5, colsample_bytree=0.5, 
               eval_metric ="rmse", scale_pos_weight=5.0)

nrounds <- 400
nfold <- 5


cv <-xgb.cv(params = params, data = trainData,nrounds = nrounds, nfold=nfold,
       metrics={'auc'}, showsd = TRUE)


#estAUC<-as.numeric(cv$test.auc.mean[nrounds])
#bst <- xgb.train(params = params, data = trainData,nrounds = 2000, 
#                 verbose=1,
#                 watchlist=list(train=trainData))
bst <- xgb.train(params = params, data = trainData,nrounds = nrounds, 
                 verbose=1,
                 watchlist=list(train=trainData))

model <- data.frame(timestamp <-Sys.time())
test_cov = subset(test_data, select=-c(bidder_id,outcome))
testMatrix <- as.matrix(test_cov)
storage.mode(testMatrix) <- "double"
pred <- predict(bst, testMatrix)
resultTest <- data.frame(bidder_id=test_data$bidder_id,prediction=pred)


