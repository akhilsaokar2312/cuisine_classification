
library(jsonlite)
train_data <- fromJSON("/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/Data/train.json", flatten=T)
test_data <- fromJSON("/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/Data/test.json", flatten=T)
library(ggplot2)
ggplot(data = train_data, aes(x = reorder(cuisine,cuisine,function(x)-length(x)))) + geom_bar() +labs(title = "Cuisine Count", x = "Cuisine", y = "Number of Recipes")

require(tm)
# If the SnowballC package is not installed and if you are trying to stem the documents then also this can occur.
# Solution: install.packages('SnowballC')
require(SnowballC)
train_ingredients <- Corpus(VectorSource(train_data$ingredients))
test_ingredients <- Corpus(VectorSource(test_data$ingredients))

# If you are doing term level transformations like tolower etc., tm_map returns character vector instead of PlainTextDocument.
# Solution: Call tolower through content_transformer or call tm_map(corpus, PlainTextDocument) immediately after tolower
train_ingredients <- tm_map(train_ingredients, content_transformer(tolower))
train_ingredients <- tm_map(train_ingredients, stripWhitespace)
train_ingredients <- tm_map(train_ingredients, removeNumbers)
train_ingredients <- tm_map(train_ingredients, removePunctuation)
train_ingredients <- tm_map(train_ingredients, stemDocument)

train_ingredientsDTM <- DocumentTermMatrix(train_ingredients)
train_ingredientsDTM_sparse <- removeSparseTerms(train_ingredientsDTM, 0.99)
train_ingredientsDTM_sparse <- as.data.frame(as.matrix(train_ingredientsDTM_sparse))
## Add the dependent variable to the data.frame
train_ingredientsDTM_sparse$cuisine <- as.factor(train_data$cuisine)

require(caret)
inTrain <- createDataPartition(y = train_ingredientsDTM_sparse$cuisine, p = 0.7, list = FALSE)
trainingDTM <- train_ingredientsDTM_sparse[inTrain,]
validatingDTM <- train_ingredientsDTM_sparse[-inTrain,]

test_ingredients <- tm_map(test_ingredients, content_transformer(tolower))
test_ingredients <- tm_map(test_ingredients, stripWhitespace)
test_ingredients <- tm_map(test_ingredients, removeNumbers)
test_ingredients <- tm_map(test_ingredients, removePunctuation)
test_ingredients <- tm_map(test_ingredients, stemDocument)

test_ingredientsDTM <- DocumentTermMatrix(test_ingredients)
# test_ingredientsDTM_sparse <- removeSparseTerms(test_ingredientsDTM, 0.99)
# test_ingredientsDTM_sparse <- as.data.frame(as.matrix(test_ingredientsDTM_sparse))
test_ingredientsDTM <- as.data.frame(as.matrix(test_ingredientsDTM))

##### ------------------- Create CART Model -------------------- #######
require(rpart)
set.seed(1234)
cartModelFit <- rpart(cuisine ~ ., data = trainingDTM, method = "class")
## Plot the tree
require(rpart.plot)
prp(cartModelFit)

cartPredict <- predict(cartModelFit, newdata = validatingDTM, type = "class")
cartCM <- confusionMatrix(cartPredict, validatingDTM$cuisine)
require(MASS)
cartPredictDataFrame <- as.data.frame(cbind(test_data$id,as.character(cartPredict)))
colnames(cartPredictDataFrame) <- c("id", "cuisine")
write.table(format(cartPredictDataFrame, scientific=FALSE), file = "/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/Submissions/cart_submission.csv", sep=",",quote = F, row.names=F)

##### ------------- Create Random Forest Model --------------- #######
require(randomForest)
rf_fit<-randomForest(cuisine~.,data=trainingDTM,ntree=1000)
rf_predictions<-predict(rf_fit,newdata=validatingDTM)
rf_CM <- confusionMatrix(rf_predictions, validatingDTM$cuisine)
rf_DataFrame <- as.data.frame(cbind(test_data$id,as.character(rf_predictions)))
colnames(rf_DataFrame) <- c("id", "cuisine")
write.table(format(rf_DataFrame, scientific=FALSE), file = "/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/Submissions/rf_submission.csv", sep=",",quote = F, row.names=F)

##### ------------- Create Gradient Boosting Model --------------- #######
require(gbm)
gbm_fit<-gbm(cuisine~.,data=trainingDTM,n.trees=1000,ty)
gbm_predictions<-predict(gbm_fit,newdata=validatingDTM,n.trees=1000)
rf_CM <- confusionMatrix(gbm_predictions, validatingDTM$cuisine)
rf_DataFrame <- as.data.frame(cbind(test_data$id,as.character(rf_predictions)))
colnames(rf_DataFrame) <- c("id", "cuisine")
write.table(format(rf_DataFrame, scientific=FALSE), file = "/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/Submissions/rf_submission.csv", sep=",",quote = F, row.names=F)

rf.model <- randomForest(cuisine~., data = trainingDTM, importance=TRUE, ntree=100, mtry = 4)

# Define the range of values over which we would want to cross-validate our model
rf.grid <-  expand.grid( n.trees = c(100), interaction.depth = c(10) , shrinkage = 0.2)
# Define the parameters for cross validation
fitControl <- trainControl(method = "none", classProbs = TRUE)
GBMmodel <- train(cuisine~., data = trainingDTM, method = "gbm", trControl = fitControl, verbose = TRUE, tuneGrid = rf.grid, metric = "ROC")

rf.pred  <- predict(rf.model, validatingDTM, type="prob")
GBMpredTrain <- predict(GBMmodel, newdata = validatingDTM, type="prob")

probs <- 0.9*rf.pred + 0.1*GBMpredTrain
final.pred <- as.factor(colnames(probs)[max.col(probs)])

class <- as.character(final.pred)
class <- as.numeric(substr(class,2,2))

results <- data.frame(Id=test$Id,Cover_Type=class)


write.csv(results, "ensembleRFGBM.csv", row.names=FALSE)

