# Recommender System Project

# The following packages are needed to run the project's code.
library(tidyverse)
library(caret)
library(lubridate)
library(knitr)
library(recosystem)
library(data.table)
library(stringr)



# The movielens dataset is downloaded and movies names and ratings are attached to it. 
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")
rm(movies, ratings)


# A Validation set will be created. It will be 10% of MovieLens data. this validation set will only be used for final assessment.
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.2, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(test_index, temp, removed, movielens)


# The loss function used to assess the accuracy of this system will be the Root Mean Square Error "RMSE".
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Exploratory Analysis
str(edx)

# The movieId variable is stored as a numeric vector. Integers are more memory efficient but they only hold whole number values. A test to check if all values are whole numbers will be made.
identical(round(edx$movieId, digits = 0), edx$movieId)

# All values are whole numbers, the column can safely be converted into class integer without changing the actual values.

# Genres variable seems to hold multiple genres, most likely ordered by relevance.
summary(str_count(edx$genres, pattern = "\\|") + 1) 
# By counting the separator "|" and adding 1, we can calculate how many parts does the genres variable have.
# How many unique genres in the variable
length(unique(edx$genres))

# Timestamp variable will be converted to a separate date and time columns to ease further analysis. The title variable holds the year each movie was made. Movie age in years at 2019 will calculated. 
# The genres variable will be split into 5 different columns while keeping the original variable to study the effect of levels reduction on variability in the average rating. The movieId variable will be converted to an integer vector.  
# Any values beyond the fifth part of the genres variable will be omitted.

edx <- edx %>% 
  separate(col = genres, into = c("genre1","genre2","genre3","genre4", "genre5"), 
           sep = "\\|", remove = FALSE) %>%
  mutate(movieId = as.integer(movieId),
         year = str_extract(title, pattern = "\\(\\d{4}\\)"),
         year = str_replace(year, "\\(", ""),
         year = str_replace(year, "\\)", ""),
         year = as.numeric(year),
         age = 2019 - year,
         date = as_datetime(timestamp)) %>%
  separate(col = date, into = c("date", "time"), sep = "\\s")

# Explore the effect of reducing the number of parts of genres on variability measured in SDs.
gen <- edx %>% select(genres, rating) %>% 
  group_by(genres) %>%
  summarise(avg = mean(rating))
df <- tibble(Method = "Genres",
             Sd = sd(gen$avg), dim = length(gen$avg))

gen <- edx %>% select(genre1, genre2, genre3, genre4, genre5, rating) %>%
  group_by(genre1, genre2, genre3, genre4, genre5) %>%
  summarise(avg = mean(rating))
df <- rbind(df,tibble(Method = "5 Genres", 
                      Sd = sd(gen$avg), dim = length(gen$avg)))

gen <- edx %>% select(genre1, genre2, genre3, genre4, rating) %>% 
  group_by(genre1, genre2, genre3, genre4) %>%
  summarise(avg = mean(rating))
df <- rbind(df,tibble(Method = "4 Genres", 
                      Sd = sd(gen$avg), dim = length(gen$avg)))

gen <- edx %>% select(genre1, genre2, genre3, rating) %>% 
  group_by(genre1, genre2, genre3) %>%
  summarise(avg = mean(rating))
df <- rbind(df,tibble(Method = "3 Genres", 
                      Sd = sd(gen$avg), dim = length(gen$avg)))

gen <- edx %>% select(genre1, genre2, rating) %>% 
  group_by(genre1, genre2) %>%
  summarise(avg = mean(rating))
df <- rbind(df,tibble(Method = "2 Genres",
                      Sd = sd(gen$avg), dim = length(gen$avg)))

gen <- edx %>% select(genre1, rating) %>% 
  group_by(genre1) %>%
  summarise(avg = mean(rating))
df <- rbind(df,tibble(Method = "1 Genre",
                      Sd = sd(gen$avg), dim = length(gen$avg)))
kable(df)


df %>% ggplot(aes(dim, Sd, color = Method)) + geom_point()

# When using only the first 2 parts in the genres variable,   


paste("dimension reduciton ", round(100 - df[5,3]/df[1,3]*100), "%")
paste("loss in variablity ", round(100 - df[5,2]/df[1,2]*100),"%")

# The first 2 parts of the genres variable will be kept. The rest will be omitted

edx <- edx %>% select(-c(genre3,genre4, genre5, genres)) %>%
  unite(genres,genre1, genre2, sep = "-", na.rm = TRUE)

# Edx will be split into test and train sets.
set.seed(2)
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

# save edx and validation sets in HDD for final validation
save(edx, validation, file = "edx_val.rda")
# remove edx and validation sets from RAM
rm(edx, validation)

# Model 1
mu <- mean(train_set$rating) 

# Calculating the standardized average for each movie.
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_1 <- RMSE(predicted_ratings, test_set$rating)


rmse_results <- tibble(Method="Movie Effect Model",
                       RMSE = model_1)

kable(rmse_results)


# Explore model's worse predictions.
# Create an error variable
t <- test_set %>% 
  mutate(pred = predicted_ratings,error = abs(rating - pred)) %>%
  group_by(title) %>% mutate(n = n()) %>% ungroup()

# plot mean error against number of ratings.   
t %>% group_by(n) %>%
  summarise(error = mean(error)) %>%
  arrange(desc(error))%>% ggplot(aes(n, error)) + 
  geom_point() + geom_smooth(method = "loess")


# Effect of using a cut off on rmse.
low_n_rmse <- t %>% filter(n < 20) %>% summarise(RMSE_low_n = sqrt(mean(error^2)))
high_n_rmse <- t %>% filter(n > 20) %>% summarise(RMSE_high_n = sqrt(mean(error^2)))
print(c(low_n_rmse,high_n_rmse))



# split the train set further into a tuning set "for cross validation" and a slightly smaller train set. 
set.seed(3)
tune_index <- createDataPartition(train_set$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set2 <- train_set[-tune_index,]
temp <- train_set[tune_index,]

tune_set <- temp %>% 
  semi_join(train_set2, by = "movieId") %>%
  semi_join(train_set2, by = "userId")

removed <- anti_join(temp, tune_set)
train_set2 <- rbind(train_set2, removed)

rm(tune_index, temp, removed)

# Loop will try different regularization parameters and outputs RSMEs
reg_par_2 <- seq(0, 5, 0.25)
model_2_rmses <- lapply(reg_par_2, function(p){
  movie_avgs <- train_set2 %>% 
    group_by(movieId) %>% 
    summarize(b_m = sum(rating - mu)/(n()+p))
  
  predicted_ratings <- mu + tune_set %>% 
    left_join(movie_avgs, by='movieId') %>%
    .$b_m
  RMSE(predicted_ratings, tune_set$rating)
})

# The regularization parameter with the lowest RMSE is.
p_m <- reg_par_2[which.min(model_2_rmses)]
p_m
plot(model_2_rmses, reg_par_2)

#Regularized movieID based model.
r_movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(r_b_m = sum(rating - mu)/(n()+p_m))

predicted_ratings <- mu + test_set %>% 
  left_join(r_movie_avgs, by='movieId') %>%
  .$r_b_m

model_2 <- RMSE(predicted_ratings, test_set$rating)



rmse_results <- rbind(rmse_results, tibble(Method="Movie Effect Model Regularized",
                                           RMSE = model_2))

kable(rmse_results)


# Loop will try different regularization parameters and outputs RSMEs. This time for userIds.
reg_par_3<- seq(3, 8, 0.25)

model_3_rmses <- lapply(reg_par_3, function(p){
  
  user_avgs <- train_set2 %>% 
    left_join(r_movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - r_b_m)/(n()+p))
  
  predicted_ratings <- tune_set %>% 
    left_join(r_movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    mutate(pred = mu + r_b_m + b_u) %>%
    .$pred
  RMSE(predicted_ratings, tune_set$rating)
})

# The best parameter is.
p_u <- reg_par_3[which.min(model_3_rmses)]
p_u


plot(model_3_rmses, reg_par_3)

r_user_avgs <- train_set %>%
  left_join(r_movie_avgs, by='movieId') %>%
  group_by(userId) %>% 
  summarize(r_b_u = sum(rating - mu - r_b_m)/(n()+p_u))

predicted_ratings <- 
  test_set %>% 
  left_join(r_movie_avgs, by = "movieId") %>%
  left_join(r_user_avgs, by = "userId") %>%
  mutate(pred = mu + r_b_m + r_b_u) %>%
  .$pred

model_3 <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- rbind(rmse_results, 
                      tibble(Method="Movie + User Effect Model Regularized",
                             RMSE = model_3))

kable(rmse_results)

# This model will add the average of each genre.

reg_par_4 <- seq(500, 2000, 100)

model_4_rmses <- lapply(reg_par_4, function(p){
  
  genres_avgs <- train_set2 %>% 
    left_join(r_movie_avgs, by='movieId') %>%
    left_join(r_user_avgs, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - r_b_m - r_b_u)/(n()+p))
  
  predicted_ratings <- tune_set %>% 
    left_join(r_movie_avgs, by='movieId') %>%
    left_join(r_user_avgs, by='userId') %>%
    left_join(genres_avgs, by='genres') %>%
    mutate(pred = mu + r_b_m + r_b_u + b_g) %>%
    .$pred
  RMSE(predicted_ratings, tune_set$rating)
})

# The best parameter is.
p_g <- reg_par_4[which.min(model_4_rmses)]
p_g


plot(model_4_rmses, reg_par_4)


# Regularized movieID + userID + genres based model. 
r_genres_avgs <- train_set %>%
  left_join(r_movie_avgs, by='movieId') %>%
  left_join(r_user_avgs, by='userId') %>%
  group_by(genres) %>% 
  summarize(r_b_g = sum(rating - mu - r_b_m - r_b_u)/(n()+p_g))

predicted_ratings <- 
  test_set %>% 
  left_join(r_movie_avgs, by = "movieId") %>%
  left_join(r_user_avgs, by = "userId") %>%
  left_join(r_genres_avgs, by = "genres") %>%
  mutate(pred = mu + r_b_m + r_b_u + r_b_g) %>%
  .$pred

model_4 <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- rbind(rmse_results, 
                      tibble(Method="Movie + User + Genres Effect Model Regularized",
                             RMSE = model_4))

kable(rmse_results)


# Model 5
# On this model, the age variable will be introduced.
t <- train_set %>% select(age, rating) %>% 
  group_by(age) %>% summarise(n = n(),avg = mean(rating)) %>% filter(n > 1000)

sd(t$avg)

t %>% ggplot(aes(age,avg)) + geom_point() + geom_smooth(method = "loess")

# Older movies have higher average ratings than newer ones. Model 5 is going to try to improve accuracy by incorporating age averages.
# This loop will find the best parameter for the age variable.
reg_par_5 <- seq(100, 150, 5)

model_5_rmses <- lapply(reg_par_5, function(p){
  
  age_avgs <- train_set2 %>% 
    left_join(r_movie_avgs, by='movieId') %>%
    left_join(r_user_avgs, by='userId') %>%
    left_join(r_genres_avgs, by='genres') %>%
    group_by(age) %>%
    summarize(b_a = sum(rating - mu - r_b_m - r_b_u - r_b_g)/(n()+p))
  
  predicted_ratings <- tune_set %>% 
    left_join(r_movie_avgs, by='movieId') %>%
    left_join(r_user_avgs, by='userId') %>%
    left_join(r_genres_avgs, by='genres') %>%
    left_join(age_avgs, by='age') %>%
    mutate(pred = mu + r_b_m + r_b_u + r_b_g + b_a) %>%
    .$pred
  RMSE(predicted_ratings, tune_set$rating)
})

# The best parameter is.
p_a <- reg_par_5[which.min(model_5_rmses)]
p_a


plot(model_5_rmses, reg_par_5)


# Regularized movieID + userID + genres + age based model.
r_age_avgs <- train_set %>%
  left_join(r_movie_avgs, by='movieId') %>%
  left_join(r_user_avgs, by='userId') %>%
  left_join(r_genres_avgs, by='genres') %>%
  group_by(age) %>% 
  summarize(r_b_a = sum(rating - mu - r_b_m - r_b_u - r_b_g)/(n()+p_a))

predicted_ratings <- 
  test_set %>% 
  left_join(r_movie_avgs, by = "movieId") %>%
  left_join(r_user_avgs, by = "userId") %>%
  left_join(r_genres_avgs, by = "genres") %>%
  left_join(r_age_avgs, by = "age") %>%
  mutate(pred = mu + r_b_m + r_b_u + r_b_g + r_b_a) %>%
  .$pred

model_5 <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- rbind(rmse_results, 
                      tibble(Method="Movie + User + Genres + Age Effect Model Regularized",
                             RMSE = model_5))

kable(rmse_results)


# Recosystem
# all not needed objects will be removed from RAM to clear space.
rm(list=ls()[! ls() %in% c("rmse_results","train_set","test_set")])

# Data transformation is needed in order to fit the package input criteria.
# Package takes in a dataframe with 2 integer columns "item and user Ids" and one numeric "rating".
r_train_set <- train_set %>% select(userId, movieId, rating)
r_test_set <- test_set %>% select(userId, movieId)

# Data will be stored in Hard drive to save RAM.
write.table(r_train_set , file = "r_train_set.txt" , sep = " ", 
            row.names = FALSE, col.names = FALSE)
write.table(r_test_set, file = "r_test_set.txt" , sep = " ",
            row.names = FALSE, col.names = FALSE)

# Create data links
r_train_set <- data_file("r_train_set.txt", package = "recosystem", index1 = TRUE)
r_test_set <- data_file("r_test_set.txt",  package = "recosystem", index1 = TRUE)

# Binding function will be used in tuning, training and prediction process. 
r <- Reco()

# Warning : Tuning process can take a very long time to run.

set.seed(4)
r_tune <- r$tune(r_train_set, opts = list(dim = c(20L, 25L),
                                          costp_l1 = 0,
                                          costp_l2 = c(0.01,0.1),
                                          costq_l1 =0,
                                          costq_l2 = c(0.01,0.1),
                                          lrate = 0.1,
                                          nthread = 4,
                                          niter = 20))

# Tuning parameters with the least RMSE will be used to train the model.
set.seed(5)
r_train <- r$train(r_train_set, opts = c(r_tune$min, niter = 30))


# The r$predict function will use the trained model to predict ratings on the test set
pred <- r$predict(r_test_set, out_memory())


model_6 <- RMSE(test_set$rating, pred)


rmse_results <- rbind(rmse_results, 
                      tibble(Method = "Recosystem",
                             RMSE = model_6))

kable(rmse_results)

# Explore recosystem prediction accuracy
# create an error variable
test_set$pred <- pred
test_set$error <- abs(test_set$pred - test_set$rating)

# The effect of the number of ratings per user and per movie on the accuracy of model 6 will be explored

# plot error against the number of times each movie was rated and the number of times each user rated.
test_set %>% group_by(movieId) %>% mutate(n_rate_movie = n()) %>% 
  ungroup() %>% group_by(n_rate_movie) %>% summarise(error = mean(error)) %>%
  arrange(desc(n_rate_movie))%>% ggplot(aes(n_rate_movie, error)) + 
  geom_point() + geom_smooth(method = "loess")

test_set %>% group_by(userId) %>% mutate(n_rate_user = n()) %>% ungroup() %>% 
  group_by(n_rate_user) %>% summarise(error = mean(error)) %>%
  arrange(desc(n_rate_user))%>% ggplot(aes(n_rate_user, error)) + 
  geom_point() + geom_smooth(method = "loess")


# Project Validation
# Clean RAM
rm(list=ls()[! ls() %in% c("r_tune","r")])

# Reload edx and validation sets
load("edx_val.rda")

# Prepare edx and validation set for Recosystem.
r_edx <- edx %>% select(userId, movieId, rating) 

r_validation <- validation %>% mutate(movieId = as.integer(movieId)) %>%
  select(userId, movieId)

# Write transformed sets to HDD
write.table(r_edx , file = "r_edx.txt" , sep = " ", 
            row.names = FALSE, col.names = FALSE)
write.table(r_validation, file = "r_validation.txt" , sep = " ",
            row.names = FALSE, col.names = FALSE)

# Create links
r_edx <- data_file("r_edx.txt", package = "recosystem", index1 = TRUE)
r_validation <- data_file("r_validation.txt",  package = "recosystem", index1 = TRUE)

# remove edx and keep validation set to calculate final RMSE
rm(edx)

# The tuning step will not be repeated, the same parameters from testing will be used.
set.seed(6)
r_train <- r$train(r_edx, opts = c(r_tune$min, niter = 30))

# predict validation set ratings.
pred <- r$predict(r_validation, out_memory())

# Check final RMSE
RMSE(validation$rating, pred)


