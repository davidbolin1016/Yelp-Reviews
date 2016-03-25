# Predicting the Usefulness of Yelp Reviews

library(dplyr)
library(stringr)
library(tm)
library(quanteda)
library(ggplot2)
library(caret)
library(glmnet)
library(dummies)
library(randomForest)

set.seed(0) # Ensure that my results can be reproduced

reviews_set1 <- read.csv("reviews01.csv")
reviews_set2 <- read.csv("reviews02.csv")
names(reviews_set2) <- names(reviews_set1)
reviews_all <- rbind(reviews_set1, reviews_set2)

# Useful columns: text of review, number of stars, number of votes that the review was useful
reviews_newdf <- select(reviews_all, stars, votes_useful, text)

# Take the number of stars as a factor
stars_df <- as.data.frame(dummy(reviews_newdf$stars))
colnames(stars_df) <- c("stars1", "stars2", "stars3", "stars4", "stars5")

# Function to clean a list of reviews
clean_text <- function(rev_list) {
  clntxt <- lapply(rev_list, as.character)
  clntxt <- tolower(clntxt)
  clntxt <- removeWords(clntxt, stopwords("SMART"))
  clntxt <- lapply(clntxt, str_replace_all, pattern = "[[:punct:]]", replacement = "")
  clntxt <- as.character(clntxt)
  return(clntxt)
}

# Clean the reviews and add the number of words for each review
reviews_newdf$text <- clean_text(reviews_newdf$text)
reviews_newdf$n_words <- sapply(reviews_newdf$text, str_count, pattern = " ") + 1
reviews_text <- reviews_newdf$text

reviews_length <- cbind(select(reviews_newdf, votes_useful, n_words)
                        , stars_df[,c(1, 2, 4, 5)]) # leaves out "3 stars" column to avoid collinearity
reviews_length$votes_useful <- log(reviews_length$votes_useful + 1)
reviews_length$n_words <- scale(log(reviews_length$n_words))
scaled_center <- attributes(reviews_length$n_words)$"scaled:center"
scaled_scale <- attributes(reviews_length$n_words)$"scaled:scale"

ggplot(reviews_length, aes(n_words, votes_useful)) + 
  geom_jitter(alpha = 0.02, width = 0, height = .9) + geom_smooth() +
  labs(title = "Length vs Usefulness", x = "Number of Words (transformed log scale)", 
       y = "Number of Votes (log scale)")

# Create dummy variables for under 20 words and over 700 words
reviews_length$under20 <- as.numeric(reviews_newdf$n_words < 20)
reviews_length$over700 <- as.numeric(reviews_newdf$n_words > 700)
l_model <- lm(votes_useful ~ ., data = reviews_length)
summary(l_model)

f_matrix <- dfm(reviews_text, ngrams = 1)
f_matrix_trimmed <- f_matrix[, colSums(f_matrix) > 800]
f_matrix_trimmed <- cbind(f_matrix_trimmed, as.matrix(reviews_length[c("n_words", "stars1",
                          "stars2", "stars4", "stars5", "under20", "over700")]))

ntest <- 111260 # takes the first set of reviews as the training set
target <- reviews_length$votes_useful
model <- cv.glmnet(f_matrix_trimmed[1:ntest,], target[1:ntest])
coefs <- coef(model, model$lambda.min)
coefs <- as.data.frame(as.matrix(coefs))
coefs <- cbind(rownames(coefs), coefs)
coefs <- coefs[coefs$"1" != 0 ,]

test_correlation <- cor(target[(ntest+1):length(target)], 
                        predict(model, f_matrix_trimmed[(ntest+1):length(target)], s = model$lambda.min)) 

# Select reviews for a particular business
sel_business <- function(df, businessid) {
  filter(df, business_id == businessid)
}

# Calculate predictions for a particular business 
new_predict <- function(business_df) {
  names(business_df) <- names(reviews_set1)
  business_df$text <- clean_text(business_df$text)
  new_matrix <- dfm(business_df$text, verbose = FALSE)
  new_matrix_names <- colnames(new_matrix)
  additional_names <- setdiff(rownames(coefs), new_matrix_names)
  additional_matrix <- Matrix(0, nrow = nrow(new_matrix), ncol = length(additional_names), sparse = TRUE)
  colnames(additional_matrix) <- additional_names
  new_matrix <- cbind(new_matrix, additional_matrix)
  new_matrix <- new_matrix[,rownames(coefs)]
  new_matrix <- as.data.frame(new_matrix)
  new_matrix$n_words <- sapply(business_df$text, str_count, pattern = " ")
  business_df$n_words <- new_matrix$n_words
  new_matrix$n_words <- scale(log(new_matrix$n_words + 1), center = scaled_center, scale = scaled_scale)
  new_matrix$stars1 <- as.numeric(business_df$stars == 1)
  new_matrix$stars2 <- as.numeric(business_df$stars == 2)
  new_matrix$stars4 <- as.numeric(business_df$stars == 4)
  new_matrix$stars5 <- as.numeric(business_df$stars == 5)
  new_matrix$under20 <- as.numeric(business_df$n_words < 20)
  new_matrix$over700 <- as.numeric(business_df$n_words > 700)
  
  # Multiply out the coefficients to calculate new predictions
  new_predictions <- rep(coefs$"1"[1], nrow(new_matrix))
  multiplied_coef <- as.matrix(new_matrix) %*% coefs[[2]]
  new_predictions <- new_predictions + rowSums(multiplied_coef)
  new_predictions <- exp(new_predictions) - 1
  new_predictions[new_predictions < 0] <- 0
  return(new_predictions)
}

# Particular example from new set of reviews
new_reviews_df <- read.csv("reviews03.csv") # A third set of reviews, outside the training and test sets
names(new_reviews_df) <- names(reviews_set1)
new_reviews <- sel_business(new_reviews_df, "lAher1puKzN9r3LALx-JqQ") # Example business
new_reviews$predictions <- new_predict(new_reviews)

# Create data frame ordered according to predicted usefulness
new_reviews_sorted <- new_reviews[order(new_reviews$predictions, decreasing = TRUE),]
new_reviews_sorted$text <- as.character(new_reviews_sorted$text)
cat(str_wrap(c("1.", new_reviews_sorted$text[1])))

new_reviews_sorted <- new_reviews[order(new_reviews$predictions),]
new_reviews_sorted$text <- as.character(new_reviews_sorted$text)
cat("1.", new_reviews_sorted$text[1])
cat("2.", new_reviews_sorted$text[2])

suppressWarnings(cor.test(new_reviews$votes_useful, new_reviews$predictions, 
                          method = "spearman", alternative = "greater"))

new_reviews_df$business_id <- as.character(new_reviews_df$business_id)
ids <- filter(new_reviews_df %>% group_by(business_id) %>% summarise(number = n()), number > 20)
cor_values <- rep(0, length(ids))
count <- 0

for (i in ids$business_id) {
  business_df <- sel_business(new_reviews_df, i)
  predictions <- new_predict(business_df)
  c = suppressWarnings(cor.test(business_df$votes_useful, predictions, 
                                method = "spearman", alternative = "greater"))
  count <- count + 1
  cor_values[count] <- c$estimate
}
cat("Average accuracy in rank ordering: ", mean(cor_values, na.rm = TRUE))

