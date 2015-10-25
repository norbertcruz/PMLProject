# Practical Machine Learning Prediction Assignment

# Load Libraries

library(caret)
library(gbm)
library(randomForest)


# Load Data

trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainUrl), na.strings = c("NA", "", "#DIV/0!"))

testing <- read.csv(url(testUrl), na.strings = c("NA", "", "#DIV/0!"))


# Set seed

set.seed(24)


# Partitioning training set

inTrain <- createDataPartition(y = training$classe, p = 2/3, list = FALSE)

train <- training[inTrain, ]

test <- training[-inTrain, ]


# Preprocessing 

nearZero <- nearZeroVar(train, saveMetrics = TRUE)

train <- train[, nearZero$nzv == FALSE]


train <- train[, -(1:6)]


countNA <- sapply(training, function(x) {sum(is.na(x))})

table(countNA)

naColumns <- names(countNA[countNA != 0])

train <- train[, !names(train) %in% naColumns]


# Apply preprocessing to test and testing sets

preProc1 <- colnames(train)

preProc2 <- colnames(train[, -53]) # remove classe var

test <- test[preProc1]

testing <- testing[preProc2]  # we only need predictors to get the answers


# Prediction Models

    # Generalized Boosted Regression

    control <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
    
    gbmFit <- train(classe ~ ., data = train, method = "gbm", trControl = control,
                    verbose = FALSE)
    
    # Random Forests
    
    rfFit <- randomForest(classe ~ ., data = train)
    

# Selecting Best Model
    
gbmPrediction <- predict(gbmFit, newdata = test)

gbmCM <- confusionMatrix(gbmPrediction, test$classe)


rfPrediction <- predict(rfFit, newdata = test, type = "class")

rfCM <- confusionMatrix(rfPrediction, test$classe)
    

# Predict Results on Testing set and write results to text file

testingPrediction <- predict(rfFit, newdata = testing, type = "class")

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(testingPrediction)