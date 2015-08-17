library(caret)
library(caretEnsemble)
library(lattice)
library(mlbench)
library(doParallel)

getModelName<-function(modelSpec) {
    paste(unlist(modelSpec), sep='_', collapse="_")
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

greedOptGini <- function(X, Y, iter = 100L){
    
    N           <- ncol(X)
    weights     <- rep(0L, N)
    pred        <- 0 * X
    sum.weights <- 0L
    
    while(sum.weights < iter) {
        
        sum.weights   <- sum.weights + 1L
        pred          <- (pred + X) * (1L / sum.weights)
        errors        <- apply(pred, MARGIN=2,function(x) {NormalizedGini(Y,x)})
        best          <- which.max(errors)
        weights[best] <- weights[best] + 1L
        pred          <- pred[, best] * sum.weights
    }
    return(weights)
}

printf <- function(...) invisible(print(sprintf(...)))

train_data<- read.csv('data/train.csv', header=TRUE)
test_data <- read.csv('data/test.csv', header=TRUE)
#train_set <- subset(train_data,select=-c(Id,T1_V4,T1_V5,T1_V15,T1_V16, T1_V11))
set.seed(37)
trainIdx <-createDataPartition(train_data$Hazard, p=0.1, list=FALSE)
train_set <- subset(train_data,select=-c(Id))[trainIdx,]

giniSummary <-function (data, lev = NULL, model = NULL) {

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
    
    c(gini=NormalizedGini(data$obs,data$pred))
}


set.seed(37)
trControl <- trainControl(method="cv",
                          number=10,
                          verboseIter=TRUE,
                          classProbs = FALSE,
                          summaryFunction = giniSummary,
                          savePredictions=TRUE,
                          index = createFolds(train_set$Hazard,k=10, list=TRUE, returnTrain=TRUE),
                          allowParallel = FALSE
)

caretModelSpecEx<-function(method,tuneGrid,...) {
    l <- apply(tuneGrid,MARGIN=1, FUN=function(x) {caretModelSpec(method, tuneGrid=as.data.frame(as.list(x)),...)})
    names(l) <-apply(tuneGrid, MARGIN=1, function(r) {paste(c('rf', r), collapse='_' )})
    l
}

model_specs <- c(list(
#    rf5=caretModelSpec(method='rf', tuneGrid=data.frame(mtry=5), ntree=50),
   # rf10=caretModelSpec(method='rf', tuneGrid=data.frame(mtry=10), ntree=150),        
   # rf20=caretModelSpec(method='rf', tuneGrid=data.frame(mtry=20), ntree=150),
#    rf21=caretModelSpec(method='rf', tuneGrid=data.frame(mtry=20), ntree=51),    
    rf30=caretModelSpec(method='rf', tuneGrid=data.frame(mtry=30), ntree=100)        
#    gamSpline10=caretModelSpec(method='gamSpline', tuneGrid = data.frame(df=10)),
#    gamSpline20=caretModelSpec(method='gamSpline', tuneGrid = data.frame(df=20)),
#    gamSpline30=caretModelSpec(method='gamSpline', tuneGrid = data.frame(df=30)),      
#    gamSpline31=caretModelSpec(method='gamSpline', tuneGrid = data.frame(df=31)),          
#    gamSpline50=caretModelSpec(method='gamSpline', tuneGrid = data.frame(df=50)),              
##    gamSpline100=caretModelSpec(method='gamSpline', tuneGrid = data.frame(df=100)),              
    #lasso0.9=caretModelSpec(method='lasso', tuneGrid = data.frame(fraction=0.9)),      
#    lasso0.91=caretModelSpec(method='lasso', tuneGrid = data.frame(fraction=0.91)),      
#    lasso0.5=caretModelSpec(method='lasso', tuneGrid = data.frame(fraction=0.5)),      
#    lasso0.2=caretModelSpec(method='lasso', tuneGrid = data.frame(fraction=0.2)),
##    gamLoees_1=caretModelSpec(method='gamLoess', tuneGrid = data.frame(span = .5, degree = 1)),
#SEGV    gamLoees_2=caretModelSpec(method='gamLoess', tuneGrid = data.frame(span = .6, degree = 2)),
#    gamLoees_2=caretModelSpec(method='gamLoess', tuneGrid = data.frame(span = .5, degree = 2)),
#    xgbTree=caretModelSpec(method='xgbTree', tuneGrid = data.frame(nrounds = 500, max_depth = 5, eta=0.001)),
##    xgbTree=caretModelSpec(method='xgbTree', tuneGrid = data.frame(nrounds = 500, max_depth = 5, eta=0.01)),
#    xgbTree=caretModelSpec(method='xgbTree', tuneGrid = data.frame(nrounds = 1000, max_depth = 10, eta=0.001)),
#    xgbTree=caretModelSpec(method='xgbTree', tuneGrid = data.frame(nrounds = 1000, max_depth = 10, eta=0.005)),
#    xgbTree=caretModelSpec(method='xgbTree', tuneGrid = data.frame(nrounds = 1000, max_depth = 15, eta=0.005))
#toolong    krlsPoly2_01=caretModelSpec(method='krlsPoly', tuneGrid = data.frame(degree=2,lambda=0.01))        
#    krlsPoly2_001=caretModelSpec(method='krlsPoly', tuneGrid = data.frame(degree=2,lambda=0.001)),        
#    krlsPoly3_01=caretModelSpec(method='krlsPoly', tuneGrid = data.frame(degree=3,lambda=0.01)),        
#    krlsPoly3_001=caretModelSpec(method='krlsPoly', tuneGrid = data.frame(degree=3,lambda=0.001))           
#    brnn20=caretModelSpec(method='brnn', tuneGrid = data.frame(neurons = 3))
    #rvmPoly=caretModelSpec(method='rvmPoly', tuneGrid = data.frame(degree = 2, scale=0.1)),
    ),
    caretModelSpecEx(method='xgbTree', tuneGrid = expand.grid(nrounds = c(500,1000,2000), max_depth = c(1,5,10), eta=c(0.1,0.01,0.001)))
)

y<-train_set$Hazard
dummies <- dummyVars(~ ., data = train_set[,-1], fullRank=TRUE)
x_train = predict(dummies, newdata = train_set[,-1])
x_test = predict(dummies, newdata = test_data[,-1])


cl <- makeCluster(10, outfile="")
registerDoParallel(cl)

model_list <- caretList(x_train,
    y,
    trControl=trControl,
    metric='gini',
    tuneList=as.list(model_specs)
)

stopCluster(cl)
#model_list$brnn20$pred$pred <-as.numeric(model_list$brnn20$pred$pred)

#xyplot(resamples(model_list))
#modelCor(resamples(model_list))

greedy_ensemble <- caretEnsemble(model_list, optFUN=greedOptGini)
summary(greedy_ensemble)
evalEnsemble(greedy_ensemble, NormalizedGini)
test_pred <-predict(greedy_ensemble,newdata=x_test)
result <- data.frame(Id=test_data$Id, Hazard=test_pred)
write.csv(result, file.path('output','submission.csv'), row.names=FALSE, quote=FALSE)

