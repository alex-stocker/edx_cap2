##########################################################################
##
#### Capstone project WineData 
#### Author: alex-stocker
##
##########################################################################

#### Project description
# I am working with wine quality data on wine quality taken from https://archive.ics.uci.edu/ml/datasets/wine+quality
# I will use  the white wine dataset that includes 11 input variables based on objective chemical tests, fixed acidity, 
# volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, 
# sulphates, # and alcohol, and one output variable, (wine) quality, a score between 1 and 10 based on the median of at 
# least 3 evaluations given by wine experts. The aim of my capstone project is to investigate machine learning approaches 
# that can predict wine quality from the input variables with the highes possible accuracy.
#### 

##########################################################################
##
#### Data preparation part 
##
##########################################################################

# Checking libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# Loading libraries
library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(gridExtra)
library(rpart.plot)
library(ggcorrplot)
library(e1071)

# I read the wine data from the data repository and rename some columns to avoid spaces..
winedata <- read_delim("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",delim=";")

# I rename the column names to avoid working with ` `.
winedata <- winedata %>% rename(fixed_acidity = `fixed acidity`,
                                volatile_acidity = `volatile acidity`,
                                citric_acid = `citric acid`,
                                residual_sugar = `residual sugar`,
                                free_so2 = `free sulfur dioxide`,
                                total_so2 = `total sulfur dioxide`)


# I create a train set and a test set.
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y=winedata$quality, times = 1, p = 0.8, list = FALSE) 
train_set <- winedata[test_index, ]
test_set <- winedata[-test_index, ]

# Function to compute the RMSE for vectors of quality ratings and their corresponding predictors
# to rate the quality of the developed models
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

##########################################################################
##
#### Data exploretion part 
##
##########################################################################

## The goal is to predict subjective  wine quality as rated by the experts from the measured parameters. 
## Before I start with modeling, I want to get better insights into the dataset, first.

## I perform some data exploration actions to get an overview on the distribution of the variables
head(train_set)
dim(train_set)
str(train_set)
summary(train_set)

# I  check the dataset for NAs
colSums(is.na(train_set)) #check for NAs

# I plot the wine quality variable
table(train_set$quality)
train_set %>% 
            ggplot(aes(x=factor(quality))) +
            geom_bar()+
            scale_x_discrete("Wine quality (between 1 and 10; 10 is best")


# I write a function to plot all input variables and look at their distribution using to 
# avoid code repetition.
plot_w <- function (input, bin){
  train_set %>% ggplot() +
    geom_histogram(aes(x={{input}}, fill=factor(quality)), binwidth={{bin}}) +
    geom_vline(aes(xintercept=mean({{input}})),linetype="dashed") +
    labs(fill = "Wine quality")
}

# Executing the plotting function
plot_w(fixed_acidity, 0.5)
plot_w(volatile_acidity, 0.01)
plot_w(citric_acid, 0.05)
plot_w(residual_sugar, 2)
plot_w(chlorides, 0.005)
plot_w(free_so2, 5)
plot_w(total_so2, 5)
plot_w(density, 0.001)
plot_w(pH, 0.02)
plot_w(sulphates, 0.02)
plot_w(alcohol, 0.2)

# I plot all input variables at once in a grid
p1 <- train_set %>% ggplot() +
  geom_histogram(aes(x=fixed_acidity))
p2 <- train_set %>% ggplot() +
  geom_histogram(aes(x=volatile_acidity))
p3 <- train_set %>% ggplot() +
  geom_histogram(aes(x=citric_acid))
p4 <- train_set %>% ggplot() +
  geom_histogram(aes(x=residual_sugar))
p5 <- train_set %>% ggplot() +
  geom_histogram(aes(x=chlorides))
p6 <- train_set %>% ggplot() +
  geom_histogram(aes(x=free_so2))
p7 <- train_set %>% ggplot() +
  geom_histogram(aes(x=total_so2))
p8 <- train_set %>% ggplot() +
  geom_histogram(aes(x=density))
p9 <- train_set %>% ggplot() +
  geom_histogram(aes(x=pH))
p10 <- train_set %>% ggplot() +
  geom_histogram(aes(x=sulphates))
p11 <- train_set %>% ggplot() +
  geom_histogram(aes(x=alcohol))
require(gridExtra)
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, ncol=3)

# Linear correlation of input factors with wine quality
correlationMatrix <- cor(train_set[,1:12])
print(correlationMatrix)

# I plot the linear correlations as heatmap
ggcorrplot(correlationMatrix)
# I compute correlations
cor_mat <- cor(as.numeric(train_set$quality), train_set[,1:11], method = "spearman")
cor_mat

# I plot the variables with highest positive/negative linear correlations
# Plot alcohol and wine quality
train_set %>% ggplot() +
  geom_boxplot(aes(x=factor(quality), y=alcohol))+
  scale_x_discrete("Wine quality")

# Plot density and wine quality
train_set %>% ggplot() +
  geom_boxplot(aes(x=factor(quality), y=density))+
  scale_x_discrete("Wine quality")

##########################################################################
##
#### Machine Learning Part 
##
##########################################################################

## In the second part, I focus on designing the models for predicting wine quality.
## Thereby I use the train_set. I test the models on the test_set computing accuracy
## and RMSE score.

#############################
# Simple model: Median model
#############################
table(train_set$quality)
median(train_set$quality)
# My simplest modek is to set all values to the median of the quality ratings
quality_pred_simplemodel <- test_set$quality
quality_pred_simplemodel[1:length(quality_pred_simplemodel)] <- median(train_set$quality)
# Compute RMSE
RMSE_simple_model <- RMSE(test_set$quality, quality_pred_simplemodel)
RMSE_simple_model
# Compute confusion matrix
# Note: Data must be factor with same levels as input for confusion matrix function
cm_simplemodel <- confusionMatrix(data = factor(quality_pred_simplemodel, levels = c(3,4,5,6,7,8,9)), 
               reference = factor(test_set$quality, levels = c(3,4,5,6,7,8,9)))
cm_simplemodel$overall["Accuracy"] 

#############################
# Linear model
#############################
# I compute a linear model using all measure values as predictors and quality as outcome.
lm_model <- lm(quality ~ ., data=train_set)
summary(lm_model) 
# Note: The "no information rate" score gives the score of the simple model (from above).
quality_pred_lm <- predict(lm_model, test_set)
# Compute RMSE
RMSE_lm <- RMSE(test_set$quality, quality_pred_lm)
RMSE_lm
# Compute confusion matrix
quality_pred_lm <- round(quality_pred_lm) ## I need to round to the nearest value (integer)
cm_lm_model <- confusionMatrix(data = factor(quality_pred_lm, levels = c(3,4,5,6,7,8,9)), 
                reference = factor(test_set$quality, levels = c(3,4,5,6,7,8,9)))
cm_lm_model$overall["Accuracy"]

#############################
# (Generalized) linear model
#############################
# I compute a glm model using all measure values as predictors and quality as outcome.
glm_model <- train(quality ~ ., data=train_set, method="glm")
summary(glm_model) 
quality_pred_glm <- predict(glm_model, test_set) 
# Compute RMSE
RMSE_glm <- RMSE(test_set$quality, quality_pred_glm)
RMSE_glm
# Compute confusion matrix
quality_pred_glm <- round(quality_pred_glm)
cm_glm_model <- confusionMatrix(data = factor(quality_pred_glm, levels = c(3,4,5,6,7,8,9)),
                reference= factor(test_set$quality, levels = c(3,4,5,6,7,8,9)))
cm_glm_model$overall["Accuracy"]

#############################
# KNN model
#############################
# I compute a KNN model using all measure values as predictors and quality as outcome.
knn_model <- train(quality ~. , data = train_set, method = "knn")
summary(knn_model)
quality_pred_knn <- predict(knn_model, test_set, type = "raw") 
# Compute RMSE
RMSE_knn <- RMSE(test_set$quality, quality_pred_knn)
RMSE_knn
# Compute confusion matrix
quality_pred_knn <- round(quality_pred_knn)
cm_knn_model <- confusionMatrix(data = factor(quality_pred_knn, levels = c(3,4,5,6,7,8,9)),
                reference= factor(test_set$quality, levels = c(3,4,5,6,7,8,9)))
cm_knn_model$overall["Accuracy"]
ggplot(knn_model, highlight = TRUE)

#############################
# Decision tree model
#############################
# I compute a decision tree model using all measure values as predictors and quality as outcome.
rpart_model <- rpart(quality ~ ., data=train_set)
summary(rpart_model)
quality_pred_rpart <- predict(rpart_model, test_set)
# Compute RMSE
RMSE_rpart <- RMSE(test_set$quality, quality_pred_rpart)
RMSE_rpart
# Compute confusion matrix
quality_pred_rpart <- round(quality_pred_rpart)
cm_rpart_model <- confusionMatrix(data = factor(quality_pred_rpart, levels = c(3,4,5,6,7,8,9)),
                reference= factor(test_set$quality, levels = c(3,4,5,6,7,8,9)))
cm_rpart_model$overall["Accuracy"]
rpart.plot(rpart_model)

#############################
# Random forest model
#############################
# I compute a random forest model using all measure values as predictors and quality as outcome.
randomForest_model <- randomForest(quality ~ ., data=train_set)
summary(randomForest_model)
quality_pred_randomForest <- predict(randomForest_model, test_set)
# Compute RMSE
RMSE_randomForest <- RMSE(test_set$quality, quality_pred_randomForest)
RMSE_randomForest
# Compute confusion matrix
quality_pred_randomForest <- round(quality_pred_randomForest)
cm_randomForest_model <- confusionMatrix(data = factor(quality_pred_randomForest, levels = c(3,4,5,6,7,8,9)),
                      reference= factor(test_set$quality, levels = c(3,4,5,6,7,8,9)))
cm_randomForest_model$overall["Accuracy"]
plot(randomForest_model)

# RF tuning
# The RF optimization leads to slightly improved model performance, but it takes very long to run,
# even with very small values for ntree and nSamp. Therefore I commented it out in my code. In theory
# further optimisation potentials are available.
# Model tuning
# control <- trainControl(method="cv", number = 5)
# grid <- data.frame(mtry = c(1:11))
# train_rf <-  train(quality ~ ., 
#                    method = "rf", 
#                    ntree = 150,
#                    trControl = control,
#                    tuneGrid = grid,
#                    nSamp = 100,
#                    data=train_set)
# randomForest_model_opt <- randomForest(quality ~ ., 
#                        minNode = train_rf$bestTune$mtry,
#                        data=train_set)
# # Fit optimised model
# quality_pred_randomForest_opt <- predict(fit_rf, test_set)
# # Compute RMSE
# RMSE_randomForest_opt <- RMSE(test_set$quality, quality_pred_randomForest)
# RMSE_randomForest_opt
# # Compute confusion matrix
# quality_pred_randomForest_opt <- round(quality_pred_randomForest)
# cm_randomForest_model_opt <- confusionMatrix(data = factor(quality_pred_randomForest, levels = c(3,4,5,6,7,8,9)),
#                                          reference= factor(test_set$quality, levels = c(3,4,5,6,7,8,9)))
# cm_randomForest_model_opt$overall["Accuracy"]

#############################
# SVM model
#############################
# I compute a SVM model using all measure values as predictors and quality as outcome.
svm_model <- svm(quality ~ ., data=train_set)
quality_pred_svm <- predict(svm_model, test_set)
# Compute RMSE
RMSE_svm <- RMSE(test_set$quality, quality_pred_svm)
RMSE_svm
# Compute confusion matrix
quality_pred_svm <- round(quality_pred_svm)
summary(svm_model)
cm_svm_model <- confusionMatrix(data = factor(quality_pred_svm, levels = c(3,4,5,6,7,8,9)),
                reference= factor(test_set$quality, levels = c(3,4,5,6,7,8,9)))
cm_svm_model$overall["Accuracy"]


##########################################################################
##
#### Final results and summary 
##
##########################################################################

# I build a results dataframe for all models evaluated based on accuracy
acc_results <- data.frame(
  Method=c("Median model", 
           "Linear model", 
           "Generalized Linar model", 
           "KNN model",
           "Decision Tree model",
           "Random Forest model",
           "SVM model"),
  Accuracy=c(cm_simplemodel$overall["Accuracy"], 
             cm_glm_model$overall["Accuracy"],
             cm_lm_model$overall["Accuracy"],
             cm_knn_model$overall["Accuracy"],
             cm_rpart_model$overall["Accuracy"],
             cm_randomForest_model$overall["Accuracy"],
             cm_svm_model$overall["Accuracy"])
             )
# I print the table
acc_results %>% arrange(desc(Accuracy))

# I build another results dataframe for all models evaluated based on RMSE score
RMSE_results <- data.frame(
  Method=c("Median model", 
           "Linear model", 
           "Generalized linear model", 
           "KNN model",
           "Decision Tree model",
           "Random Forest model",
           "SVM model"),
  RMSE=c(RMSE_simple_model, 
             RMSE_lm,
             RMSE_glm,
             RMSE_knn,
             RMSE_rpart,
             RMSE_randomForest,
             RMSE_svm)
)
# I print the table
RMSE_results %>% arrange(RMSE)

## I summarise my first KPI for the model (highest accuracy)
# highest accuracy
acc_best <- acc_results %>% 
  arrange(desc(Accuracy)) %>% 
  filter(row_number()==1) %>% 
  pull(Accuracy)
# best model
acc_best_model <- acc_results %>% 
  arrange(desc(Accuracy)) %>% 
  filter(row_number()==1) %>% 
  pull(Method)

acc_best_model
acc_best

## I summarise my second KPI for the model (lowest RMSE score)
# Lowest RMSE score
RMSE_best <- RMSE_results %>% 
  arrange(RMSE) %>% 
  filter(row_number()==1) %>% 
  pull(RMSE)
# Lowest best model
RMSE_best_model <- RMSE_results %>% 
  arrange(RMSE) %>% 
  filter(row_number()==1) %>% 
  pull(Method)

RMSE_best_model 
RMSE_best 

##
## I finally completed my script with showing the best model, highest accuracy, and lowest
## RMSE score. 
##