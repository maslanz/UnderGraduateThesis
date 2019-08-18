library(tidyverse)
library(caret)
library(data.table)
library(pROC)
library(Matrix)
library(MatrixModels)
library(kernlab)
library(party)

dataset <- read.csv("adult_census_income.csv")

numeric_variables <- c('age', 'fnlwgt', 'education.num',
                       'capital.gain', 'hours.per.week', 'capital.loss')
dataset[numeric_variables] <- map(dataset[numeric_variables], ~ as.numeric(.))

dataset %>%
  mutate(income = case_when(
    income == "<=50K" ~ "low",
    income == ">50K" ~ "high",
    TRUE ~ "Unk"
  ))  -> dataset

set.seed(0)
in_tr <- createDataPartition(dataset$income, p=.7, list=FALSE)

train_set <- dataset[in_tr,]
test_set <- dataset[-in_tr,]

train_x = train_set[, -15]
train_y = train_set[, 15]
test_x = test_set[, -15]
test_y = test_set[, 15]

#seyrek matris
train_x_trans <- as.matrix(sparse.model.matrix(~ ., data = train_x))
test_x_trans <- as.matrix(sparse.model.matrix(~ ., data = test_x))

#
is_zv <- nearZeroVar(train_x_trans)
train_x_trans <- train_x_trans[, -is_zv]
test_x_trans <- test_x_trans[, -is_zv]

ctrl <- trainControl(method="cv", n=5, summaryFunction=twoClassSummary, 
                     savePredictions=TRUE, classProbs=TRUE)

test_y = factor(test_y, levels = c('high', 'low'))

# Logistic Regression
set.seed(0)
glmFit <- train(x = train_x_trans,
                y = train_y,
                method = "glm",
                metric = "ROC",
                trControl = ctrl)

glmFit

glmROC <- roc(response = glmFit$pred$obs,
              predictor = glmFit$pred$high,
              levels = rev(levels(glmFit$pred$obs)))

# Linear Discriminant Analysis
set.seed(0)
ldaFit <- train(x = train_x_trans,
                y = train_y,
                method = "lda",
                trControl = ctrl,
                metric = "ROC")

ldaFit

ldaROC <- roc(response = ldaFit$pred$obs,
    predictor = ldaFit$pred$high,
    levels = rev(levels(ldaFit$pred$obs)))

# K-Nearest Neighboorhood (KNN)
set.seed(0)
knnFit <- train(income ~.,
                data = train_set,
                method = "knn",
                metric = "ROC",
                trControl = ctrl)

knnFit

knnROC <- roc(response = knnFit$pred$obs,
              predictor = knnFit$pred$high,
              levels = rev(levels(knnFit$pred$obs)))

# Bagged Tree
set.seed(0)
baggedFit <- train(income ~.,
                   data = train_set,
                   method = "treebag",
                   metric = "ROC",
                   trControl = ctrl)

baggedFit

bagROC <- roc(response = baggedFit$pred$obs,
              predictor = baggedFit$pred$high,
              levels = rev(levels(baggedFit$pred$obs)))

# Boosted Tree
set.seed(0)
gbmFit <- train(income ~.,
                data = train_set,
                method = "gbm",
                metric = "ROC",
                verbose = FALSE,
                trControl = ctrl)

gbmFit

gbmROC <- roc(response = gbmFit$pred$obs,
              predictor = gbmFit$pred$high,
              levels = rev(levels(gbmFit$pred$obs)))

# Random Forest
set.seed(0)
rfFit <- train(income ~.,
               data = train_set,
               method = "rf",
               ntree = 100,
               metric = "ROC",
               trControl = ctrl)

rfFit
plot(rfFit)

rfROC <- roc(response = rfFit$pred$obs,
             predictor = rfFit$pred$high,
             levels = rev(levels(rfFit$pred$obs)))


# ---*Multiple and One-by-One ROC Curves*---
visRoc <- ggroc(list(RF = rfROC, LDA = ldaROC, 
                     LogReg = glmROC, Bagged = bagROC,
                     KNN = knnROC, GBM = gbmROC),
                legacy.axes = TRUE, aes = c("color")) +
  theme_bw() +
  labs(x = "False Positive Rate",
       y = "True Positive Rate",
       title = "ROC Comparison for Various Models") +
  geom_line(size = .75, alpha = 2/3)  

visRoc

# Predicting the Test Set and Metric Comparisons
glmPred <- predict(glmFit, newdata = test_x_trans, type = "raw")
ldaPred <- predict(ldaFit, newdata = test_x_trans, type = "raw")
rfPred <- predict(rfFit, newdata = test_set[, -15], type = "raw")
bagPred <- predict(baggedFit, newdata = test_set[, -15], type = "raw")
gbmPred <- predict(gbmFit, newdata = test_set[, -15], type = "raw")
knnPred <- predict(knnFit, newdata = test_set[, -15], type = "raw")

postResults <- data.frame(
  LogisticReg = postResample(glmPred, test_y),
  LDA = postResample(ldaPred, test_y),
  BaggedTrees = postResample(bagPred, test_y),
  RandomForests = postResample(rfPred, test_y),
  BoostedTrees = postResample(gbmPred, test_y),
  KNN = postResample(knnPred, test_y))

postResults <- tidyr::gather(postResults, model, rate)
postResults$type <- c('Accuracy', 'Kappa')
str(postResults)

postResults$model <- as.factor(postResults$model)
postResults$rate <- as.numeric(postResults$rate)
postResults$type <- as.factor(postResults$type)

postResults$rate <- as.numeric(round(postResults$rate, 4))

postResults

postResults1 <- postResults %>%
  filter(type == "Accuracy")

vis1 <-  ggplot(postResults1, aes(x = reorder(model, rate), y = rate)) +
  theme_bw() +
  expand_limits(y = c(0, 1.0)) +
  scale_y_continuous(breaks = seq(0, 1, by = .1)) +
  geom_col(position = "dodge", fill = "firebrick",
           width = .5) +
  
  labs(title = "Accuracy Rate of Models",
       subtitle = "Head-to-Head Comparison",
       x = "Model",
       y = NULL) +
  coord_flip()

vis1

# end of the project...

save.image("tez2.RData")