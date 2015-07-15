library(caret)

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

printf <- function(...) invisible(print(sprintf(...)))

train_data<- read.csv('data/train.csv', header=TRUE)
test_data <- read.csv('data/test.csv', header=TRUE)
#train_set <- subset(train_data,select=-c(Id,T1_V4,T1_V5,T1_V15,T1_V16, T1_V11))
trainIdx <-createDataPartition(train_data$Hazard, p=0.1, list=FALSE)

train_set <- subset(train_data,select=-c(Id))#[trainIdx,]

set.seed(37)

giniSummary <-function (data, lev = NULL, model = NULL) {
    #print(head(data))
    c(gini=NormalizedGini(data$obs,data$pred))
}

trControl <- trainControl(method="cv",
                          number=5,
                          verboseIter=TRUE,
                          classProbs = FALSE,
                          summaryFunction = giniSummary
)
#                         allowParallel = FALSE)

tuneGrid <- data.frame(mtry=c(20,25))

model <- train(Hazard ~ ., data = train_set, 
            method="rf",
            ntree=50,
            importance = FALSE,
            trControl = trControl,
            tuneGrid = tuneGrid,
            metric = "gini",
            maximize = TRUE         
)

model$timestamp <-Sys.time()
bestTune <- model$results[order(as.numeric(row.names(model$result)))[as.numeric(row.names(model$bestTune))],]
print("Best:")
bestTune


test_pred <-predict(model,newdata=test_data)
result <- data.frame(Id=test_data$Id, Hazard=test_pred)
write.csv(result, "submission.csv", row.names=FALSE, quote=FALSE)

