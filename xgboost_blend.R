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
modelsFiles <- dir('target/task', pattern='pred-.*')
modelIds <-sapply(strsplit(modelsFiles, '-'), function(x){x[2]})
modelIds <- modelIds[order(modelIds)]

input <-lapply(modelIds, function(f) { load(file.path('target', 'task', paste('pred',f, sep='-'))); cv$pred })
names(input) <- modelIds
input <- data.frame(input)
ref <- train_data$Hazard

modelParams <- do.call("rbind", lapply(modelIds, function(f) {
    read.table(file.path('target', 'task', paste('out',f, sep='-')), sep=' ')} ))
rownames(modelParams) <- modelIds

set.seed(100)
folds <- createFolds(train_data$Hazard,k=nfold, list=TRUE, returnTrain=FALSE)

blend_ensamble <- function(data,reference,iter=500,init=10,sample=1, prune=0.02) {
#    data <-as.data.frame(lapply(data, function(l) {
#        ord <- rep(0, lenght.out =nrow(data))
#        ord[order(l)] = 1:nrow(data)
#        ord
#        }))
    #data <- as.data.frame(lapply(data, function(x) {exp(x)}))
    no_models <- ncol(data)
    #data<-as.matrix(data)
    model_num <-init
    single_model_metrics <- sapply(data, function(x){NormalizedGini(reference, x) })
    sm_desc_ord  <-order(single_model_metrics, decreasing = TRUE)
    initial_sm <- sm_desc_ord[1:model_num]
    weights <- rep(0, length.out=length(single_model_metrics))
    gini <-rep(0,length.out=iter)
    
    # assign initial weights
    weights[initial_sm] <- 1
    sums <- apply(data[, initial_sm], 1,sum)
    avg_prediction <- sums/model_num
    avg_gini <-NormalizedGini(reference, avg_prediction)
    print("Initial gini: ")
    print(avg_gini)
    gini[1] = avg_gini
    
    for (i in 2:iter ) {
        model_num <- model_num + 1
        metrics <- sapply(input, function(x) {
            if (runif(1)< sample) {
                SumModelGini(reference, (x + sums)/model_num) 
            } else {
                0
            }
        })
        sel <- which.max(metrics)
        weights[sel] <- weights[sel] + 1
        sums <- sums + data[,sel]
        avg_prediction <- sums/model_num
        avg_gini <-NormalizedGini(reference, avg_prediction)
        print(i)
        print(single_model_metrics[sel])
        print(avg_gini)
        gini[i] = avg_gini
    }
    
    #prune 
    pruned_weights <- weights;
    pruned_weights[pruned_weights<prune*length(pruned_weights)] =0
    avg_pred <- apply(data, MARGIN=1, FUN=function(x) {sum(x*pruned_weights)/sum(pruned_weights)} )
    
    list(weights=weights, gini=gini, metric=avg_gini, pruned_weights=pruned_weights, 
         pruned_gini=NormalizedGini(reference, avg_pred), single_model_metrics = single_model_metrics )
}


evalEnsamble <- function(weights, labels, predictions, useFolds = NULL) {
    normalizedWeights <- weights/sum(weights)
    ensamblePrediction <- apply(predictions, MARGIN=1, FUN=function(r) { sum(normalizedWeights*r)})
    if (is.null(useFolds)) {
        NormalizedGini(labels, ensamblePrediction)
    } else {
        mean(sapply(folds, function(f){ NormalizedGini(labels[f], ensamblePrediction[f]) }))
    }
}


res <- blend_ensamble(input, ref, iter=200, sample=0.5, prune=0.01)
colnames(modelParams) <- c('train.gini.mean', 'train.gini.sd', 'test.gini.mean', 'test.gini.sd', 'nrounds', 
    'max.depth', 'eta', 'subsample', 'colsample_bytree', 'min_child_weight','gamma')
ensambleParams <-data.frame(modelParams, weights=res$weights, pruned_weights=res$pruned_weights, 
                            single_model_metrics = res$single_model_metrics)
write.csv(ensambleParams, file.path('target','ensamble.csv'), quote=FALSE)

