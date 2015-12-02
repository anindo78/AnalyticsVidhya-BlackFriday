setwd("~/Kaggle/AV-Black Friday")
library(caret)
library(data.table)

# read train data
train <- fread("train.csv", header=T)



# replacing missing product_id with 9999
train$Product_Category_1[is.na(train$Product_Category_1)] <- 9999
train$Product_Category_2[is.na(train$Product_Category_2)] <- 9999
train$Product_Category_3[is.na(train$Product_Category_3)] <- 9999

#creating key for prod cat
train$hier <- paste(train$Product_Category_1,"-", train$Product_Category_2,"-",train$Product_Category_3)

unique_prod <- train[, .(cnt_unique_prod=sum(!duplicated(Product_ID))),by=.(hier)]
unique_prod

prod_sum <- train[, .(totsales=sum(Purchase),totcnt=.N, avgsales=mean(Purchase)),by=.(Product_ID)]
prod_sum

prod_sum$prob_pur <- (prod_sum$totcnt/550068)
quantile(prod_sum$prob_pur, probs=seq(0,1,0.1))

prod_sum$exp_sales <- prod_sum$prob_pur * prod_sum$avgsales
quantile(prod_sum$exp_sales, probs=seq(0,1,0.1))

prod_sum

product <- prod_sum[, .(Product_ID, exp_sales)]

#merge
train0 <- merge(train, product, by="Product_ID")
head(train0)


length(unique(train$Product_ID))
#3631
length(unique(train$Product_Category_1))
length(unique(train$Product_Category_2))
length(unique(train$Product_Category_3))


# deselecting user_id and product_id
train1 <- train0[, c("Product_ID", "User_ID","hier"):=NULL]



train1 <- as.data.frame(train1)
#convert them into factor vars
train1$Occupation <- as.factor(train1$Occupation)
train1$Marital_Status <- as.factor(train1$Marital_Status)
train1$Product_Category_1 <- as.factor(train1$Product_Category_1)
train1$Product_Category_2 <- as.factor(train1$Product_Category_2)
train1$Product_Category_3 <- as.factor(train1$Product_Category_3)
train1$Gender <- as.factor(train1$Gender)
train1$Age <- as.factor(train1$Age)
train1$City_Category <- as.factor(train1$City_Category)
train1$Stay_In_Current_City_Years <- as.factor(train1$Stay_In_Current_City_Years)
train1$City_Category <- as.factor(train1$City_Category)

#creating dummy vars
train2 <- predict(dummyVars(~ ., data = train1), newdata = train1)
train2 <- as.data.frame(train2)

#scale the exp_sales between 0 and 1
min <- min(train2$exp_sales)
max <- max(train2$exp_sales)
#sd <- sd(train2$Purchase)

train2$purchase_scale <- (train2$exp_sales - min) / (max-min)
train2$exp_sales <- NULL
# p <- train2$Purchase
# mean <- mean(p)
# sd <- sd(p)
# train2$scale_pur <- (p - mean)/sd

# split train into test and train
bound <- floor((nrow(train2)/4)*3)         #define % of training and test set

set.seed(101)
train2 <- train2[sample(nrow(train2)), ]           #sample rows 
dftrain <- train2[1:bound, ]              
dftest <- train2[(bound+1):nrow(train2), ]


# runnign regression
y <- as.vector(log(dftrain$Purchase))
x <- as.matrix(dftrain[, -c(95)])

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
                  maxit = 500,
                  ## and the number of parameters used by the model
                  MaxNWts=5* (ncol(Xn) + 1) + 5 + 1)

# Testing Model on Train

dftrain$pred_nn_purchase <- predict(nnetAvg, Xn)

dftrain$log_purchase <- log(dftrain$Purchase)

cor <- cor(dftrain$pred_nn_purchase, dftrain$log_purchase)
cor
# 0.876914
Rsq <- cor ^ 2
Rsq
# 0.7689782


# Testing Model on Test
xx <- as.matrix(dftest[, -c(95)])

Xk <- xx[, names(as.data.frame(xx)) %in% names(as.data.frame(Xn))]

dftest$pred_nn_purchase <- predict(nnetAvg, Xk)
dftest$log_purchase <- log(dftest$Purchase)

cor <- cor(dftest$pred_nn_purchase, dftest$log_purchase)
cor
# 0.876914
Rsq <- cor ^ 2
Rsq
# 0.7689782

# RMSE
dftest$pred_purchase <- exp(dftest$pred_nn_purchase)
RMSE <- sqrt(sum((dftest$Purchase-dftest$pred_purchase)^2)/nrow(dftest))
RMSE
# 2798.471


# GBM method
library(gbm)
Xn <-  as.data.frame(Xn)
gbmfit <- gbm(y ~., Xn,
                     n.trees=1000,
                     shrinkage=0.01,
                     distribution="gaussian",
                     interaction.depth=4,
                     bag.fraction=0.7,
                     cv.fold=5,
                     n.minobsinnode = 50
)

plot(gbmfit)
summary(gbmfit)


# Testing Model on Train

dftrain$pred_nn_purchase <- predict(gbmfit, Xn)

dftrain$log_purchase <- log(dftrain$Purchase)

cor <- cor(dftrain$pred_nn_purchase, dftrain$log_purchase)
cor
# 0.8692383
Rsq <- cor ^ 2
Rsq
# 0.7555753


# Testing Model on Test
xx <- as.matrix(dftest[, -c(95)])

Xk <- xx[, names(as.data.frame(xx)) %in% names(as.data.frame(Xn))]
Xk <- as.data.frame(Xk)

dftest$pred_nn_purchase <- predict(gbmfit, Xk)
dftest$log_purchase <- log(dftest$Purchase)

cor <- cor(dftest$pred_nn_purchase, dftest$log_purchase)
cor
# 0.8709752
Rsq <- cor ^ 2
Rsq
# 0.7585977

# RMSE
dftest$pred_purchase <- exp(dftest$pred_nn_purchase)
RMSE <- sqrt(sum((dftest$Purchase-dftest$pred_purchase)^2)/nrow(dftest))
RMSE
# 2944.469



# OLD METHOD
library(glmnet)
y <- as.vector(log(dftrain$Purchase))
x <- as.matrix(dftrain[, -c(95)])

tooHigh <- findCorrelation(cor(x), cutoff = .75)
Xn <- x[, -tooHigh]

# removing corr. predictors
tooHigh <- findCorrelation(cor(x), cutoff = .75)
Xn <- x[, -tooHigh]

fit <- cv.glmnet(Xn, y, alpha = 0.1)

dftrain$pred_nn_purchase <- predict(fit, Xn)

dftrain$log_purchase <- log(dftrain$Purchase)

cor <- cor(dftrain$pred_nn_purchase, dftrain$log_purchase)
cor
# 0.8688019
Rsq <- cor ^ 2
Rsq
# 0.7548167

# Testing on test data
xx <- as.matrix(dftest[, -c(95)])

Xk <- xx[, names(as.data.frame(xx)) %in% names(as.data.frame(Xn))]
Xk <- as.data.frame(Xk)

dftest$pred_nn_purchase <- predict(fit, Xk)
dftest$log_purchase <- log(dftest$Purchase)

cor <- cor(dftest$pred_nn_purchase, dftest$log_purchase)
cor
# 0.8702014
Rsq <- cor ^ 2
Rsq
# 0.7572505

# RMSE
dftest$pred_purchase <- exp(dftest$pred_nn_purchase)
RMSE <- sqrt(sum((dftest$Purchase-dftest$pred_purchase)^2)/nrow(dftest))
RMSE
# 2922.213
