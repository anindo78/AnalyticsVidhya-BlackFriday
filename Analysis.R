setwd("~/AV-Black Friday")
library(caret)


# read train data
train <- read.csv("train.csv", header=T)


# deselecting user_id and product_id
train1 <- train[, -c(1,2)]

# replacing missing product_id with 9999
train1$Product_Category_1[is.na(train1$Product_Category_1)] <- 9999
train1$Product_Category_2[is.na(train1$Product_Category_2)] <- 9999
train1$Product_Category_3[is.na(train1$Product_Category_3)] <- 9999

#convert them into factor vars
train1$Product_Category_1 <- as.factor(train1$Product_Category_1)
train1$Product_Category_2 <- as.factor(train1$Product_Category_2)
train1$Product_Category_3 <- as.factor(train1$Product_Category_3)

#creating dummy vars
train2 <- predict(dummyVars(~ ., data = train1), newdata = train1)
train2 <- as.data.frame(train2)

# runnign regression
library(glmnet)
y <- as.vector(log(train2$Purchase))
x <- as.matrix(train2[, -c(74)])

# removing corr. predictors
tooHigh <- findCorrelation(cor(x), cutoff = .75)
Xn <- x[, -tooHigh]

fit <- cv.glmnet(Xn, y, alpha = 0.1)
plot(fit)
fit$lambda.min
# 0.003070502

coef <- coef(fit, s="lambda.min")

train2$pred_purchase <- predict(fit, newx = Xn, s="lambda.min")
train2$log_purchase <- log(train2$Purchase)

cor <- cor(train2$pred_purchase, train2$log_purchase)
cor
# 0.8602599
Rsq <- cor ^ 2
Rsq
# 0.7400472

#plot error
ax <- extendrange(c(train2$log_purchase, train2$pred_purchase))
plot(train2$log_purchase, train2$pred_purchase, ylim=ax, xlim=ax)
abline(0, 1, col="darkgrey", lty=2)

train2$error <- train2$log_purchase - train2$pred_purchase
plot(train2$pred_purchase, train2$error, xlab="pred", ylab="error")

# export test data
test <- read.csv("test.csv", header=T)

# deselecting user_id and product_id
test1 <- test[, -c(1,2)]

# replacing missing product_id with 9999
test1$Product_Category_1[is.na(test1$Product_Category_1)] <- 9999
test1$Product_Category_2[is.na(test1$Product_Category_2)] <- 9999
test1$Product_Category_3[is.na(test1$Product_Category_3)] <- 9999

#convert them into factor vars
test1$Product_Category_1 <- as.factor(test1$Product_Category_1)
test1$Product_Category_2 <- as.factor(test1$Product_Category_2)
test1$Product_Category_3 <- as.factor(test1$Product_Category_3)

#creating dummy vars
test2 <- predict(dummyVars(~ ., data = test1), newdata = test1)
test2 <- as.data.frame(test2)

# insert dummy factors
test2$Product_Category_1.19 <- 0
test2$Product_Category_1.20 <- 0


str(test2)
# predict on test file
xt <- as.matrix(test2)

test$pred_log_purchase <- predict(fit, newx = xt, s="lambda.min")

test$pred_purchase <- exp(test$pred_log_purchase)
write.csv(test, file="test_final.csv")

# Neural Net Model
tooHigh <- findCorrelation(cor(x), cutoff = .75)
Xn <- x[, -tooHigh]

nnetAvg <- avNNet(Xn, y,
                  size = 5,
                  decay = 0.01,
                  ## Specify how many models to average
                  repeats = 5,
                  linout = TRUE,
                  ## Reduce the amount of printed output
                  trace = FALSE,
                  ## Expand the number of iterations to find
                  ## parameter estimates..
                  maxit = 100,
                  ## and the number of parameters used by the model
                  MaxNWts=5* (ncol(Xn) + 1) + 5 + 1)


train2$pred_nn_purchase <- predict(nnetAvg, Xn)

cor <- cor(train2$pred_nn_purchase, train2$log_purchase)
cor
# 0.8323891
Rsq <- cor ^ 2
Rsq
# 0.7400472


# Go ahead with GLMNET model


