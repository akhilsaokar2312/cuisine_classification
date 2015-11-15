library(jsonlite)
train.data <- fromJSON("/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/cuisine_classification/Data/train.json", flatten=T)
test.data <- fromJSON("/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/cuisine_classification/Data/test.json", flatten=T)
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

combined.ingredients.DTM <- DocumentTermMatrix(combined.ingredients)
combined.ingredients.sparse.DTM <- removeSparseTerms(combined.ingredients.DTM, 0.99)
combined.ingredients.sparse.DTM <- as.data.frame(as.matrix(combined.ingredients.sparse.DTM))

training.samples = dim(train.data)[1]
training.ingredients.DTM = combined.ingredients.sparse.DTM[1:training.samples,]
training.ingredients.DTM$cuisine = train.data$cuisine
require(caret)
partition.training.samples <- createDataPartition(training.ingredients.DTM$cuisine, p = 0.7, list = FALSE)
training.subset.DTM <- training.ingredients.DTM[partition.training.samples,]
validating.subset.DTM <- training.ingredients.DTM[-partition.training.samples,]

testing.samples = dim(test.data)[1]
testing.ingredients.DTM = combined.ingredients.sparse.DTM[(training.samples+1):(training.samples+testing.samples),]
##### ------------------- Create CART Model -------------------- #######
require(rpart)
cart.model.fit <- rpart(cuisine ~., data = training.ingredients.DTM, method = "class")
## Plot the tree
require(rpart.plot)
prp(cart.model.fit)

cart.model.predicted.cuisine <- predict(cart.model.fit, newdata = testing.ingredients.DTM, type = "class")
# cart.confusion.matrix <- confusionMatrix(cart.model.predicted.cuisine, validating.subset.DTM$cuisine)
require(MASS)
cart.model.output.data <- as.data.frame(cbind(test.data$id,as.character(cart.model.predicted.cuisine)))
colnames(cart.model.output.data) <- c("id", "cuisine")
write.table(format(cart.model.output.data, scientific=FALSE), file = "/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/Submissions/cart_submission.csv", sep=",",quote = F, row.names=F)

##### ------------- Create Random Forest Model --------------- #######
require(randomForest)
rf_fit<-randomForest(cuisine~.,data=training.subset.DTM,ntree=1000)
rf_predictions<-predict(rf_fit,newdata=validatingDTM)
rf_CM <- confusionMatrix(rf_predictions, validatingDTM$cuisine)
rf_DataFrame <- as.data.frame(cbind(test_data$id,as.character(rf_predictions)))
colnames(rf_DataFrame) <- c("id", "cuisine")
write.table(format(rf_DataFrame, scientific=FALSE), file = "/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/Submissions/rf_submission.csv", sep=",",quote = F, row.names=F)

