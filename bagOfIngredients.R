library(jsonlite)
train.data <- fromJSON("/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/Data/train.json", flatten=T)
test.data <- fromJSON("/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/Data/test.json", flatten=T)
combined.data.frame <- rbind(train.data[!names(train.data) %in% "cuisine"], test.data)

library(ggplot2)
ggplot(data = train.data, aes(x = reorder(cuisine,cuisine,function(x)-length(x)))) + geom_bar() +labs(title = "Cuisine Count", x = "Cuisine", y = "Number of Recipes")

require(tm)
# If you are doing term level transformations like tolower etc., tm_map returns character vector instead of PlainTextDocument.
# Solution: Call tolower through content_transformer or call tm_map(corpus, PlainTextDocument) immediately after tolower
combined.ingredients <- Corpus(VectorSource(combined.data.frame$ingredients))
combined.ingredients <- tm_map(combined.ingredients, content_transformer(tolower))
combined.ingredients <- tm_map(combined.ingredients, stripWhitespace)
combined.ingredients <- tm_map(combined.ingredients, removeNumbers)
combined.ingredients <- tm_map(combined.ingredients, removePunctuation)

# If the SnowballC package is not installed and if you are trying to stem the documents then also this can occur.
# Solution: install.packages('SnowballC')
require(SnowballC)
combined.ingredients <- tm_map(combined.ingredients, stemDocument)

# train_ingredients <- Corpus(VectorSource(train_data$ingredients))
# test_ingredients <- Corpus(VectorSource(test_data$ingredients))

# train_ingredients <- tm_map(train_ingredients, content_transformer(tolower))
# train_ingredients <- tm_map(train_ingredients, stripWhitespace)
# train_ingredients <- tm_map(train_ingredients, removeNumbers)
# train_ingredients <- tm_map(train_ingredients, removePunctuation)
# train_ingredients <- tm_map(train_ingredients, stemDocument)

combined.ingredients.DTM <- DocumentTermMatrix(combined.ingredients)
combined.ingredients.sparse.DTM <- removeSparseTerms(combined.ingredients.DTM, 0.99)
combined.ingredients.sparse.DTM <- as.data.frame(as.matrix(combined.ingredients.sparse.DTM))
## Add the dependent variable to the data.frame
combined.ingredients.sparse.DTM$cuisine <- as.factor(train_data$cuisine)

require(caret)
inTrain <- createDataPartition(y = train_ingredientsDTM_sparse$cuisine, p = 0.7, list = FALSE)
trainingDTM <- train_ingredientsDTM_sparse[inTrain,]
validatingDTM <- train_ingredientsDTM_sparse[-inTrain,]

test_ingredients <- tm_map(test_ingredients, content_transformer(tolower))
test_ingredients <- tm_map(test_ingredients, stripWhitespace)
test_ingredients <- tm_map(test_ingredients, removeNumbers)
test_ingredients <- tm_map(test_ingredients, removePunctuation)
test_ingredients <- tm_map(test_ingredients, stemDocument)

test_ingredientsDTM <- DocumentTermMatrix(test_ingredients,list(dictionary = train_ingredientsDTM_sparse))
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

