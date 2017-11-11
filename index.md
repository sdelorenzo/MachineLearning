# Coursera Practical Machine Learning Project
Stephen DeLorenzo  

## Summary

The goal of this project is to design a machine learning model to predict how well a subject is pefoming an exercise using data collected from wearable devices.  A variety of models were tested with Random Forest emerging as a highly accurate method, reaching 99% estimated out of sample accuracy.

## Exploratory Analysis/Data Cleaning

Load needed packages and read data into R.  The RandomForest algorithm is computationally expensive, so I will prep my machine to run some of the computing in parallel across cores with the parallel and doparallel packages.

```r
library(caret)
library(parallel)
library(doParallel)
   train <- read.csv("C:/Users/sdeloren/Downloads/pml-training.csv")
   test <- read.csv("C:/Users/sdeloren/Downloads/pml-testing.csv")
```

This data set has many columns with blank and null records, so I will find and eliminate them.  Each transformation done to the train set must also be done to the test set for downstream prediction.


```r
 blank <- vector()
        for (i in names(train)){
                blank[i] <- sum(is.na(train[,i]))
        }
        drop <- names(blank[blank>0])
         train2 <- train[,!(names(train) %in% drop)]
         test2 <- test[,!(names(test) %in% drop)]
```

Next test for any columns that contain little to no variance and elimiate them, as they wont help the model.  Also remove other columns that dont affect the outcome like X (row number) 

```r
novar <- nearZeroVar(train2)
  train2 <- train2[,-novar]
  test2 <- test2[,-novar]
train2 <- train2[,-c(1,3:6)]
test2 <- test2[,-c(1,3:6)]
```

Sometimes predictors in a data set are highly correlated with one another which just turns out to be redundant information to the model.  Here I search for and remove those predictors.


```r
correlations <- cor(train2[-c(1,54)])
  rm_cor <- findCorrelation(correlations,cutoff=.85,names=TRUE)
    train2 <- train2[,!(names(train2) %in% rm_cor)]
    test2 <- test2[,!(names(test2) %in% rm_cor)]
```

Before training the model, check the distribution of outcomes for potential class bias.  This data set is fairly well distributed.  We have a clean data set which is ready to test a first round model build.

```r
table(train2$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

## Model Building

Split train set further into a training/test set

```r
inTrain <- createDataPartition(train2$classe,p=.7,list=FALSE)
  sub_train <- train2[inTrain,]
  sub_test <- train2[-inTrain,]
```

The Random Forest model takes a very long time to run on this dataset, making it impractical to use.  The following allows for multi core parallel processing, as well as changes caret's normal RF behavior to 10 fold cross validation.

```r
cluster <- makeCluster(detectCores() - 1)
  registerDoParallel(cluster)
  fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
```

Then set a seed and train a Random Forest model as well asn a Linear Discriminant model for comparison.  I also attempted a Gradient Boosted model but R kept crashing.


```r
set.seed(314)
        rfmod <- train(classe~.,method="rf",data=sub_train, trControl=fitControl)
set.seed(314)
        ldmod <- train(classe~.,method="lda",data=sub_train)
```

The Random Foroest model produces excellent results with 99% in sample accuracy, where the Linear Discriminate model achieves 69% in sample accuracy.  Next to predict on our test split for an out of sample accuracy estimate.  This too comes back around 99% for Random Forest, which will constitute our final model.


```r
rfpred <- predict(rfmod,sub_test)
        confusionMatrix(rfpred,sub_test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    9    0    1    0
##          B    2 1127    9    0    0
##          C    0    3 1015    9    0
##          D    0    0    2  954    2
##          E    0    0    0    0 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9937          
##                  95% CI : (0.9913, 0.9956)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.992           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9895   0.9893   0.9896   0.9982
## Specificity            0.9976   0.9977   0.9975   0.9992   1.0000
## Pos Pred Value         0.9941   0.9903   0.9883   0.9958   1.0000
## Neg Pred Value         0.9995   0.9975   0.9977   0.9980   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1915   0.1725   0.1621   0.1835
## Detection Prevalence   0.2858   0.1934   0.1745   0.1628   0.1835
## Balanced Accuracy      0.9982   0.9936   0.9934   0.9944   0.9991
```

Last we predict on the original test data set for the quiz.

```r
finalpred <- predict(rfmod,test2)
```

## Conclusion
A Random Forest fit works well to predict the final 'classe' from this data set.  There was some preprocessing of data to remove highly correlated predictors and junk records, but the remaining predictors served well in the model so there was not a need to further preprocess such as centering/scaling.

## Appendix
In sample Accuracy

```r
rfmod
```

```
## Random Forest 
## 
## 13737 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12362, 12361, 12363, 12363, 12364, 12364, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9888620  0.9859096
##   25    0.9903183  0.9877517
##   49    0.9809265  0.9758650
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 25.
```

```r
ldmod
```

```
## Linear Discriminant Analysis 
## 
## 13737 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.6979757  0.6172982
```

Important variables

```r
varImp(rfmod)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 49)
## 
##                      Overall
## yaw_belt              100.00
## pitch_forearm          82.12
## magnet_dumbbell_z      64.16
## magnet_dumbbell_y      56.02
## roll_forearm           49.27
## magnet_belt_y          42.57
## gyros_belt_z           32.43
## magnet_belt_z          31.07
## roll_dumbbell          27.02
## accel_dumbbell_y       25.45
## magnet_dumbbell_x      24.33
## accel_forearm_x        23.29
## total_accel_dumbbell   20.70
## magnet_belt_x          20.68
## roll_arm               20.60
## accel_dumbbell_z       19.39
## magnet_forearm_z       19.15
## total_accel_belt       18.19
## yaw_arm                17.29
## accel_forearm_z        16.38
```
