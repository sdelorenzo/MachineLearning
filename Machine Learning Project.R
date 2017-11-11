library(caret)
library(parallel)
library(doParallel)
# load the data
        train <- read.csv("C:/Users/sdeloren/Downloads/pml-training.csv")
        test <- read.csv("C:/Users/sdeloren/Downloads/pml-testing.csv")

# Exploratory Analysis/Data Cleaning
        # remove columns that are mostly NA
        blank <- vector()
        for (i in names(train)){
                blank[i] <- sum(is.na(train[,i]))
        }
        drop <- names(blank[blank>0])
         train2 <- train[,!(names(train) %in% drop)]
         test2 <- test[,!(names(test) %in% drop)]
        
        # remove columns that explain little to no variance
        novar <- nearZeroVar(train2)
         train2 <- train2[,-novar]
         test2 <- test2[,-novar]
        
        #remove X (row number), duplicated timestamps and window
        train2 <- train2[,-c(1,3:6)]
        test2 <- test2[,-c(1,3:6)]
        
        # look for highly correlated predictors, there are a few so we should either use PCA or remove some more columns
        correlations <- cor(train2[-c(1,54)])
        rm_cor <- findCorrelation(correlations,cutoff=.85,names=TRUE)
        train2 <- train2[,!(names(train2) %in% rm_cor)]
        test2 <- test2[,!(names(test2) %in% rm_cor)]
        
        # check out how many records of each class there are
        table(train2$classe)
        
# Model Building
        # split train set further into a training/test set
        inTrain <- createDataPartition(train2$classe,p=.7,list=FALSE)
        sub_train <- train2[inTrain,]
        sub_test <- train2[-inTrain,]
        
        # random forest model takes a very long time to run on this dataset, the following is to allow for multi core parallel processing
        cluster <- makeCluster(detectCores() - 1)
        registerDoParallel(cluster)
        fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
        
        # fit the random forest model
        set.seed(314)
        rfmod <- train(classe~.,method="rf",data=sub_train, trControl=fitControl)
        
        # fit a linear discriminate model for comparison
        set.seed(314)
        ldmod <- train(classe~.,method="lda",data=sub_train)
        
        # gradient boosting continues to crash
        #gbmod <- train(classe~.,method="gbm",data=sub_train,verbose=FALSE, trControl=trainControl(allowParallel=TRUE))
        
        # random forest model produces excellent results with 99% in sample accuracy
        rfmod
        
        # linear discriminate model achieves 69% in sample accuracy
        ldmod
        
        rfpred <- predict(rfmod,sub_test)
        ldpred <- predict(ldmod,sub_test)
        
        confusionMatrix(rfpred,sub_test$classe)
        confusionMatrix(ldpred,sub_test$classe)
        
        # view the variable importance from rf model
        varImp(rfmod)
        
# make final prediction on test set
        finalpred <- predict(rfmod,test2)