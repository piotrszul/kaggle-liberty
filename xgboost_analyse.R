modelsFiles <- dir('target/task', pattern='pred-.*')
modelIds <-sapply(strsplit(modelsFiles, '-'), function(x){x[2]})
modelIds <- modelIds[order(modelIds)]


modelParams <- do.call("rbind", lapply(modelIds, function(f) {
    read.table(file.path('target', 'task', paste('out',f, sep='-')), sep=' ')} ))
rownames(modelParams) <- modelIds
colnames(modelParams) <- c('train.gini.mean', 'train.gini.sd', 'test.gini.mean', 'test.gini.sd', 'nrounds', 
                           'max.depth', 'eta', 'subsample', 'colsample_bytree', 'min_child_weight','gamma', 'taskid')

