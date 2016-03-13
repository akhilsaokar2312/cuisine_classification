library(jsonlite)
train.data <- fromJSON("/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/cuisine_classification/Data/train.json", flatten=T)
test.data <- fromJSON("/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/cuisine_classification/Data/test.json", flatten=T)
combined.data.frame <- rbind(train.data[!names(train.data) %in% "cuisine"], test.data)

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
training.ingredients.DTM$cuisine = as.factor(train.data$cuisine)
require(caret)
partition.training.samples <- sample(training.samples, 0.7 * training.samples)
training.subset.DTM <- training.ingredients.DTM[partition.training.samples,]
validating.subset.DTM <- training.ingredients.DTM[-partition.training.samples,]

testing.samples = dim(test.data)[1]
testing.ingredients.DTM = combined.ingredients.sparse.DTM[(training.samples+1):(training.samples+testing.samples),]

##### ------------------- Create CART Model -------------------- #######
require(rpart)
require(rpart.plot)
require(MASS)

# --- Run this block of code only during cross validation --- #
# cart.model.fit <- rpart(cuisine ~., data = training.subset.DTM, method = "class")
# prp(cart.model.fit)
# cart.model.predicted.cuisine <- predict(cart.model.fit, newdata = validating.subset.DTM, type = "class")
# cart.confusion.matrix <- confusionMatrix(cart.model.predicted.cuisine, validating.subset.DTM$cuisine)
# cart.confusion.matrix

# ------- Run this block of code only during testing -------- #
cart.model.fit <- rpart(cuisine ~., data = training.ingredients.DTM, method = "class")
prp(cart.model.fit)
cart.model.predicted.cuisine <- predict(cart.model.fit, newdata = testing.ingredients.DTM, type = "class")

cart.model.output.data <- as.data.frame(cbind(test.data$id,as.character(cart.model.predicted.cuisine)))
colnames(cart.model.output.data) <- c("id", "cuisine")
write.table(format(cart.model.output.data, scientific=FALSE), file = "/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/Submissions/cart_submission.csv", sep=",",quote = F, row.names=F)

##### ------------- Create Random Forest Model --------------- #######
require(randomForest)
# --- Run this block of code only during cross validation --- #
# rf.model.fit<-randomForest(cuisine~.,data=training.subset.DTM,ntree=200)
# rf.model.predicted.cuisine<-predict(rf.model.fit,newdata=validating.subset.DTM)
# rf.confusion.matrix <- confusionMatrix(rf.model.predicted.cuisine, validating.subset.DTM$cuisine)

# ------- Run this block of code only during testing -------- #
rf.model.fit<-randomForest(cuisine~.,data=training.ingredients.DTM,ntree=200)
rf.model.predicted.cuisine<-predict(rf.model.fit,newdata=testing.ingredients.DTM)
rf.model.output.data <- as.data.frame(cbind(test.data$id,as.character(rf.model.predicted.cuisine)))
colnames(rf.model.output.data) <- c("id", "cuisine")
write.table(format(rf.model.output.data, scientific=FALSE), file = "/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/Submissions/rf_submission.csv", sep=",",quote = F, row.names=F)

##### ------------- Create Extreme Gradient Boosting Model --------------- #######
require(xgboost)
# --- Run this block of code only during cross validation --- #
xgb.grid <- expand.grid(nround = c(25,50,100), max.depth = c(10,25), eta = c(0.4,0.6))
xgb.document.matrix <- xgb.DMatrix(as.matrix(training.subset.DTM[!names(training.subset.DTM) %in% "cuisine"]),label=as.numeric(training.subset.DTM$cuisine)-1)
for(i in 1:nrow(xgb.grid)) {
  model.xgb.train <- xgboost(data = xgb.document.matrix, nthread=3, nround =xgb.grid[i,'nround'], max.depth=xgb.grid[i,'max.depth'], eta=xgb.grid[i,'eta'], objective = "multi:softmax", verbose = 0,num_class=20)
  xgb.model.predicted.cuisine <- predict(model.xgb.train,as.matrix(validating.subset.DTM[!names(validating.subset.DTM) %in% "cuisine"]))
  xgb.model.predicted.cuisine <- factor(xgb.model.predicted.cuisine,labels=levels(training.subset.DTM$cuisine))
  xgb.confusion.matrix <- confusionMatrix(xgb.model.predicted.cuisine,validating.subset.DTM$cuisine)
  print(format(list(nthread=3, nround =xgb.grid[i,'nround'], max.depth=xgb.grid[i,'max.depth'], eta=xgb.grid[i,'eta'],accuracy = xgb.confusion.matrix$overall[1])))
}

# ------- Run this block of code only during testing -------- #
xgb.document.matrix <- xgb.DMatrix(as.matrix(training.ingredients.DTM[!names(training.ingredients.DTM) %in% "cuisine"]),label=as.numeric(training.ingredients.DTM$cuisine)-1)
model.xgb.train <- xgboost(data = xgb.document.matrix, nthread=3, nround = 50, max.depth = 10, eta=0.4, objective = "multi:softmax", verbose = 0,num_class=20)
xgb.model.predicted.cuisine <- predict(model.xgb.train,as.matrix(testing.ingredients.DTM[!names(testing.ingredients.DTM) %in% "cuisine"]))
xgb.model.predicted.cuisine <- factor(xgb.model.predicted.cuisine,labels=levels(training.ingredients.DTM$cuisine))
xgb.model.output.data <- as.data.frame(cbind(test.data$id,as.character(xgb.model.predicted.cuisine)))
colnames(xgb.model.output.data) <- c("id", "cuisine")
write.table(format(xgb.model.output.data, scientific=FALSE), file = "/Users/akhilsaokar2312/UCLA/Fall 15/MS Project/Submissions/xgb_submission.csv", sep=",",quote = F, row.names=F)

# --------- estimate variable importance -----------#
xgb.importance <- varImp(model.xgb.train, scale=FALSE)
# summarize importance
print(xgb.importance)
# plot importance
plot(xgb.importance$importance[1:20,])
text(xgb.importance$importance[1:20,],rownames(xgb.importance$importance)[1:20], offset = 0.5,pos=1)


  