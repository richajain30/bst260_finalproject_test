# Load necessary libraries
library(tm)
library(randomForest)
library(ranger)

# Create a dataframe with the provided data
test <- read.table("train.txt", sep = ";", header = FALSE)
colnames(test) <- c("Description", "Mood")
test$label <- c(1, 2, 3, 4, 5, 6)[match(test$Mood, c('sadness', 'anger', 'love', 'surprise', 'fear', 'joy'))]
head(test)

# Preprocessing - Convert text to lowercase
test$Description <- tolower(test$Description)

# Text vectorization using TF-IDF
corpus <- Corpus(VectorSource(test$Description))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
dtm <- DocumentTermMatrix(corpus)
#tfidf <- weightTfIdf(dtm)

# Combine the TF-IDF matrix with the mood labels
tfidf_df <- as.data.frame(as.matrix(dtm))
tfidf_df$Mood <- as.factor(test$Mood)

colnames(tfidf_df) <- paste(colnames(tfidf_df), "_c", sep = "")

#tfidf_df$Mood
nt <- seq(10, 150, by = 10)
oob_predictions <- vector("numeric", length(nt))

# Training the Random Forest model
for(i in 1:length(nt)){
  rf_model <- ranger::ranger(Mood_c ~ ., data = tfidf_df, num.trees = nt[i])
  oob_predictions[i] <- rf_model$prediction.error
  print(i)
}

oob_predictions <- 1 - oob_predictions # gives us the accuracy. 
max(oob_predictions)
plot(x = nt, y = oob_predictions, col = "red", type = "l")
#shows us that plateau at about 150 trees


# Example prediction
#new_text <- c("i am so so happy")
#new_text <- tolower(new_text)
#new_text <- removePunctuation(new_text)
#new_text <- removeNumbers(new_text)
#new_text <- removeWords(new_text, stopwords("en"))
#new_text_tfidf <- DocumentTermMatrix(
#  Corpus(VectorSource(new_text)),
#  control = list(dictionary = Terms(dtm))
#)
#new_text_tfidf <- as.data.frame(as.matrix((new_text_tfidf)))
#colnames(new_text_tfidf) <- paste(colnames(new_text_tfidf), "_c", sep = "")
#predicted_mood <- predict(rf_model, data = new_text_tfidf, type = "response", predict.all=TRUE)$predictions
#predicted_mood


