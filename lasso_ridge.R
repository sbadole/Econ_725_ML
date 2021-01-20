
################################################################
# Econ 725 Project Code for Lasso and Ridge and Random Forest  #
################################################################


rm(list=ls())
# Loading libraries required and reading the data into R
library(plyr)
library(dplyr)
library(corrplot)
library(caret)
library(gridExtra)
library(scales)
library(Rmisc)
library(ggrepel)
library(psych)
library(xgboost)
library(tidyverse)
library(haven)
library(haven)
library(lubridate)
library(cowplot)
library(caret)
library(leaps)
library(MASS)
library(rsample)  
library(glmnet)   
library(ggplot2)  
library(pdp)
suppressMessages(library(knitr))
suppressMessages(library(h2o))
suppressMessages(library(randomForest))
suppressMessages(library(neuralnet))
suppressMessages(library(data.table))
suppressMessages(library(stargazer))
suppressMessages(library(glmnet))
suppressMessages(library(knitr))
suppressMessages(library(tidyverse))
library(svglite)


# Load Data
setwd("/Users/sachin/Downloads/Econ_725/Projects/code")
df1 <- read_csv("cleaned_data.csv")


# Dim
dim(df1)

# # display first 10 variables and the response variable
str(df1[,c(1:10, 81)]) 

# The response variable; biddy1 (Histogram)
# ggplot(data=df1[!is.na(df1$biddy1),], aes(x=biddy1)) +
#   geom_histogram(fill="blue", binwidth = 10000) +
#   scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = comma)

# The response variable; logbid1 (Histogram)
# ggplot(data=df1[!is.na(df1$logbid1),], aes(x=logbid1)) +
#   geom_histogram(fill="blue", binwidth = 1) +
#   scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = comma)

# summary
summary(df1$logbid1)

#
df1 <- df1[!(is.na(df1$logbid1)),]
df1 <- df1[!(is.na(df1$photos)),]
df1 <- df1[!(is.na(df1$logmiles)),]
df1 <- df1[!(is.na(df1$logtext)),]

# Find most na columns
colSums(is.na(df1))

# Delete Na columns
cok=apply(df1,2,function(x)!any(is.na(x)))
df1 = df1[,cok] 


###################################################################
# Correlations with logbid1
numericVars <- which(sapply(df1, is.numeric)) #index vector numeric variables

# Numeric varibales data
data_ebay <- df1[, numericVars]

# order data
col_order <- c(383, 1:382, 384:457)
data_ebay <- data_ebay[, col_order]

# delete biddy1
data_ebay = subset(data_ebay, select = -c(biddy1, numbids, model, text, miles) )

# Summary Stat data
data_sum = subset(data_ebay, select = c(logbid1, sell, logmiles, startbid, cyl, photos, doors,
                                        rust_group, logtext, reserve, buyitnow, condition, relist) )

# save data summary in latex
library(stargazer)
stargazer(as.data.frame(data_sum), summary.stat = c("n", "mean", "sd"), type = "latex", 
          digits=1,flip = FALSE, out = 'tab.txt')

# Important variable data
data_impv = subset(data_ebay, select = c(logbid1, sell, reserve, descriptionsize, buyitnow,
                                         condition, relist, logtext, logmiles, photos, cyl, rust_group) )

# Variable Importance function
varImp <- function(object, lambda = NULL, ...) {
  
  ## skipping a few lines
  
  beta <- predict(object, s = lambda, type = "coef")
  if(is.list(beta)) {
    out <- do.call("cbind", lapply(beta, function(x) x[,1]))
    out <- as.data.frame(out, stringsAsFactors = TRUE)
  } else out <- data.frame(Overall = beta[,1])
  out <- abs(out[rownames(out) != "(Intercept)",,drop = FALSE])
  out
}


#################################################################################################
# Lasso and Ridge Model function

lasso_ridge = function(data_ebay, sell, log_bid) {
  
  if(sell == TRUE){data_ebay <- data_ebay[data_ebay$sell ==1, ]}
  else{data_ebay <- data_ebay}
  

  ####################################################################
  # Sample split
  
  set.seed(0)
  smp_size = floor(0.98*nrow(data_ebay))
  train_ind <- sample(seq_len(nrow(data_ebay)), size = smp_size)
  
  traindata = data_ebay[train_ind, ]
  testdata = data_ebay[-train_ind, ]
  
  if(log_bid == TRUE){
    data_train_x <- model.matrix(logbid1 ~ ., traindata)[, -1]
    data_train_y <- traindata$logbid1

    data_test_x <- model.matrix(logbid1 ~ ., testdata)[, -1]
    data_test_y <- testdata$logbid1
  }
  else{
    data_train_x <- model.matrix(sell ~ ., traindata)[, -1]
    data_train_y <- traindata$sell

    data_test_x <- model.matrix(sell ~ ., testdata)[, -1]
    data_test_y <- testdata$sell
  }
  
  
  ##########################################################################
  # Ridge some best model
  ##########################################################################
  
  cv_ridge   <- cv.glmnet(data_train_x, data_train_y, alpha = 0)
  min(cv_ridge$cvm)
  
  #coefficients of ridge model
  coef(cv_ridge, s=cv_ridge$lambda.1se)
  
  # predict
  pred_r <- predict(cv_ridge, s = cv_ridge$lambda.min, data_test_x)
  
  # Over all mse for ridge model
  mse_ridge <- mean((data_test_y - pred_r)^2)
  
  # RMSE for ridge model
  rmse_ridge = sqrt(mse_ridge)
  
  # Variable Importance
  var_imp = varImp(cv_ridge, lambda = cv_ridge$lambda.min)
  var_imp = as.data.frame(var_imp)
  var_imp$names <- rownames(var_imp)
  
  # Most Importance Variables
  ridgeImportance <- var_imp[order(var_imp$Overall, decreasing = TRUE),]
  
  # Variables selection
  varsSelected <- length(which(ridgeImportance$Overall!=0))
  varsNotSelected <- length(which(ridgeImportance$Overall==0))
  
  cat('Ridge uses', varsSelected, 'variables in its model, and did not select', varsNotSelected, 'variables.')
  
  # #### Top 25 influential variables in ridge (Plot)
  # coef(cv_ridge, s = "lambda.1se") %>%
  #   tidy() %>%
  #   filter(row != "(Intercept)") %>%
  #   top_n(25, wt = abs(value)) %>%
  #   ggplot(aes(value, reorder(row, value))) +
  #   geom_point() +
  #   ggtitle("Top 25 influential variables") +
  #   xlab("Coefficient") +
  #   ylab(NULL)
  # 
  # 
  ########################################################################
  # Lasso some best model
  #######################################################################
  cv_lasso   <- cv.glmnet(data_train_x, data_train_y, alpha = 1.0)
  min(cv_lasso$cvm)
  
  # Coefficients for Lasso model
  coef(cv_lasso, s=cv_lasso$lambda.1se)
  
  # Prediction from Lasso model
  pred_l <- predict(cv_lasso, s = cv_lasso$lambda.min, data_test_x)
  
  # Over all MSE for Lasso Model
  mse_lasso <- mean((data_test_y - pred_l)^2)
  
  # RMSE for Lasso model
  rmse_lasso = sqrt(mse_lasso)
  
  # Variable Importance
  var_imp1 = varImp(cv_lasso, lambda = cv_lasso$lambda.min)
  var_imp1 = as.data.frame(var_imp1)
  var_imp1$names <- rownames(var_imp1)
  
  # Most Importance Variables
  lassoImportance <- var_imp1[order(var_imp1$Overall, decreasing = TRUE),]
  
  # Variables selection
  varsSelected1 <- length(which(lassoImportance$Overall!=0))
  varsNotSelected1 <- length(which(lassoImportance$Overall==0))
  
  cat('Lasso uses', varsSelected1, 'variables in its model, and did not select', varsNotSelected1, 'variables.')
  
  #### Top 25 influential variables in Lasso (Plot)
  # coef(cv_lasso, s = "lambda.1se") %>%
  #   tidy() %>%
  #   filter(row != "(Intercept)") %>%
  #   top_n(25, wt = abs(value)) %>%
  #   ggplot(aes(value, reorder(row, value))) +
  #   geom_point() +
  #   ggtitle("Top 25 influential variables") +
  #   xlab("Coefficient") +
  #   ylab(NULL)
  

  # #########################################################################
  # results <- data.frame(Model = c('Ridge', 'Lasso'),
  #                       MSE = c(mse_ridge, mse_lasso),
  #                       RMSE = c(rmse_ridge, rmse_lasso))
  # ######################################################################### 
  #
  # Print results
  print(paste("Results for Lasso MSE", mse_lasso))
  print(paste("Results for Ridge MSE", mse_ridge))
  print(paste("Results for Lasso RMSE", rmse_lasso))
  print(paste("Results for Ridge RMSE", rmse_ridge))



  # save results in latex
  #library(stargazer)
  #stargazer(results, summary = FALSE, type = "latex", out='rlog_1.tex')
  
  
  
}


###############################################################################
# Function for Random forest
##############################################################################

random_forest = function(data_ebay, sell, log_bid) {
  
  if(sell == TRUE){data_ebay <- data_ebay[data_ebay$sell ==1, ]}
  else{data_ebay <- data_ebay}
  
  
  ####################################################################
  # Sample split
  
  set.seed(0)
  smp_size = floor(0.98*nrow(data_ebay))
  train_ind <- sample(seq_len(nrow(data_ebay)), size = smp_size)
  
  traindata = data_ebay[train_ind, ]
  testdata = data_ebay[-train_ind, ]
  
  if(log_bid == TRUE){
    data_train_x <- model.matrix(logbid1 ~ ., traindata)[, -1]
    data_train_y <- traindata$logbid1
    
    data_test_x <- model.matrix(logbid1 ~ ., testdata)[, -1]
    data_test_y <- testdata$logbid1
  }
  else{
    data_train_x <- model.matrix(sell ~ ., traindata)[, -1]
    data_train_y <- traindata$sell
    
    data_test_x <- model.matrix(sell ~ ., testdata)[, -1]
    data_test_y <- testdata$sell
  }
  
  
  # #######################################################################################
  # # # Random Forest Model using h2o
  # ########################################################################################
  # set.seed(0)
  # h2o.init()
  
  rf <- h2o.randomForest(x=2:457, y = 1, training_frame = as.h2o(traindata),
                         validation_frame = as.h2o(testdata), ntree = 50, seed = 420)
  
  
  # MSE for Random forest model
  mse_rf <- h2o.mse(rf)
  mse_rf
  
  # RMSE for Random forest model
  rmse_rf <- sqrt(mse_rf)
  
  # Varibale Importance
  imprf <- h2o.varimp(rf)
  imprf
  
  
  # Print results
  print(paste("Results for Random Forest MSE", mse_rf))
  print(paste("Results for Random Forest RMSE", rmse_rf))
  
}

################################################################################################
# Table 1: Lasso and Ridge model MSE and RMSE for (logbid1) where Sell = 0 & 1 (with full data)
lasso_ridge(data_ebay = data_ebay, sell = FALSE, log_bid = TRUE)

###############################################################################################
# Table 2: Lasso and Ridge model MSE and RMSE for (logbid1) where Sell = 1 (with full data)
lasso_ridge(data_ebay = data_ebay, sell = TRUE, log_bid = TRUE)

###############################################################################################
# Table 3: Lasso and Ridge model MSE and RMSE for Sell as dependent variable (with full data)
lasso_ridge(data_ebay = data_ebay, sell = FALSE, log_bid = FALSE)

################################################################################################
# Table 4: Lasso and Ridge model MSE and RMSE for data (logbid1) where Sell = 0 & 1 (with important variable data)
lasso_ridge(data_ebay = data_impv, sell = FALSE, log_bid = TRUE)

###############################################################################################
# Table 5: Lasso and Ridge model MSE and RMSE for data (logbid1) where Sell = 1 (with important variable data)
lasso_ridge(data_ebay = data_impv, sell = TRUE, log_bid = TRUE)

###############################################################################################
# Table 6: Lasso and Ridge model MSE and RMSE for Sell (with important variable data)
lasso_ridge(data_ebay = data_impv, sell = FALSE, log_bid = FALSE)

##############################################################################################

set.seed(0)
h2o.init()

#############################################################################################
# Table 1: Random Forest Model MSE and RMSE for (logbid1) where Sell = 0 & 1 (with full data)
random_forest(data_ebay = data_ebay, sell = FALSE, log_bid = TRUE)

############################################################################################
# Table 2: Random Forest Model MSE and RMSE for (logbid1) where Sell = 1 (with full data)
random_forest(data_ebay = data_ebay, sell = TRUE, log_bid = TRUE)

###############################################################################################
# Table 3:  Random Forest Model MSE and RMSE for Sell as dependent variable (with full data)
random_forest(data_ebay = data_ebay, sell = FALSE, log_bid = FALSE)

################################################################################################
# Table 4: Random Forest Model MSE and RMSE for data (logbid1) where Sell = 0 & 1 (with important variable data)
random_forest(data_ebay = data_impv, sell = FALSE, log_bid = TRUE)

###############################################################################################
# Table 5: Random Forest Model MSE and RMSE for data (logbid1) where Sell = 1 (with important variable data)
random_forest(data_ebay = data_impv, sell = TRUE, log_bid = TRUE)

###############################################################################################
# Table 6: Random Forest Model MSE and RMSE for Sell (with important variable data)
random_forest(data_ebay = data_impv, sell = FALSE, log_bid = FALSE)

##############################################################################################









######################################### RESULTS ###############################################

# ################################################################################################
# # Table 1: MSE and RMSE for (logbid1) where Sell = 0 & 1 (with full data)
# lasso_ridge(data_ebay = data_ebay, sell = FALSE, log_bid = TRUE)
# 
# Ridge uses 396 variables in its model, and did not select 55 variables.
# Lasso uses 348 variables in its model, and did not select 103 variables.
# [1] "Results for Lasso MSE 0.534331775326611"
# [1] "Results for Ridge MSE 0.534960024552168"
# [1] "Results for Lasso RMSE 0.73098001021"
# [1] "Results for Ridge RMSE 0.731409614752341"
# [1] "Results for Random Forest MSE 0.2661826808493"
# [1] "Results for Random Forest RMSE 0.515928949419685"

# ###############################################################################################
# # Table 2: MSE and RMSE for (logbid1) where Sell = 1 (with full data)
# lasso_ridge(data_ebay = data_ebay, sell = TRUE, log_bid = TRUE)
# 
# Ridge uses 382 variables in its model, and did not select 69 variables.
# Lasso uses 325 variables in its model, and did not select 126 variables.
# [1] "Results for Lasso MSE 0.653156953387956"
# [1] "Results for Ridge MSE 0.657424539538216"
# [1] "Results for Lasso RMSE 0.808181262705314"
# [1] "Results for Ridge RMSE 0.810817204762094"
# [1] "Results for Random Forest MSE 0.339252011583686"
# [1] "Results for Random Forest RMSE 0.582453441558796"
# ###############################################################################################
# # Table 3: MSE and RMSE for Sell as dependent variable (with full data)
# lasso_ridge(data_ebay = data_ebay, sell = FALSE, log_bid = FALSE)
# 
# Ridge uses 396 variables in its model, and did not select 55 variables.
# Lasso uses 1 variables in its model, and did not select 450 variables.
# [1] "Results for Lasso MSE 0.000187205144207223"
# [1] "Results for Ridge MSE 0.00241564707000077"
# [1] "Results for Lasso RMSE 0.0136822930902398"
# [1] "Results for Ridge RMSE 0.0491492326491551"
# ###############################################################################################
# # Table 4: MSE and RMSE for data (logbid1) where Sell = 0 & 1 (with important variable data)
# random_forest(data_ebay = data_impv, sell = FALSE, log_bid = TRUE)
# 
# Ridge uses 10 variables in its model, and did not select 0 variables.
# Lasso uses 10 variables in its model, and did not select 0 variables.
# [1] "Results for Lasso MSE 0.762542113717504"
# [1] "Results for Ridge MSE 0.763365817304146"
# [1] "Results for Lasso RMSE 0.873236573740189"
# [1] "Results for Ridge RMSE 0.873708084719459"
# [1] "Results for Random Forest MSE 0.569932362691605"
# [1] "Results for Random Forest RMSE 0.754938648296406"
# ###############################################################################################
# # Table 5: MSE and RMSE for data (logbid1) where Sell = 1 (with important variable data)
# random_forest(data_ebay = data_impv, sell = TRUE, log_bid = TRUE)
# 
# Ridge uses 9 variables in its model, and did not select 2 variables.
# Lasso uses 9 variables in its model, and did not select 2 variables.
# [1] "Results for Lasso MSE 1.02125328931025"
# [1] "Results for Ridge MSE 1.02391797648491"
# [1] "Results for Lasso RMSE 1.0105707740234"
# [1] "Results for Ridge RMSE 1.0118883221408"
# [1] "Results for Random Forest MSE 0.732866555759845"
# [1] "Results for Random Forest RMSE 0.856076255808935"
# ###############################################################################################
# # Table 6: MSE and RMSE for Sell (with important variable data)
# random_forest(data_ebay = data_impv, sell = FALSE, log_bid = FALSE)
# 
# Ridge uses 11 variables in its model, and did not select 0 variables.
# Lasso uses 1 variables in its model, and did not select 10 variables.
# [1] "Results for Lasso MSE 0.000187205144207223"
# [1] "Results for Ridge MSE 0.00219073391941099"
# [1] "Results for Lasso RMSE 0.0136822930902398"
# [1] "Results for Ridge RMSE 0.0468052766193193"
# ###############################################################################################
# 




####################################################################
# Random Forest for partial plot using 12 covariates
###################################################################
set.seed(0)
smp_size = floor(0.98*nrow(data_impv))
train_ind <- sample(seq_len(nrow(data_impv)), size = smp_size)

traindata = data_impv[train_ind, ]
testdata = data_impv[-train_ind, ]

# Random Forest model
rf1 <- randomForest(logbid1 ~ ., traindata, ntree=50, norm.votes=FALSE)

# Plot 1 Logmiles vs yhat (Prediction)
p1 <- partial(rf1 , pred.var = "logmiles", plot = FALSE)

g1 = ggplot(data=p1, aes(x=logmiles, y=yhat)) +
  geom_line(color="red")+
  geom_point() + xlab("Log miles") + ylab("Prediction") 

# Plot 2 Cylinder vs yhat (Prediction)
p2 <- partial(rf1 , pred.var = "cyl", plot = FALSE)

g2 = ggplot(data=p2, aes(x=cyl, y=yhat)) +
  geom_line(color="red")+
  geom_point() + xlab("cylinders") + ylab("Prediction") 

# Plot 3 Log text vs yhat (Predication)
p3 <- partial(rf1 , pred.var = "logtext", plot = FALSE)

g3 = ggplot(data=p3, aes(x=logtext, y=yhat)) +
  geom_line(color="red")+
  geom_point() + xlab("Log text") + ylab("Prediction") 

# Plot 4 Photos vs yhat (Prediction)
p4 <- partial(rf1 , pred.var = "photos", plot = FALSE)

g4 = ggplot(data=p4, aes(x=photos, y=yhat)) +
  geom_line(color="red")+
  geom_point() + xlab("Photos") + ylab("Prediction") 

# Plot all four graphs
f1 <- grid.arrange(g1, g2, g3, g4, ncol = 2)  

# Save plot in svg formate
ggsave(file="rf_plot_0.svg", plot= f1, width=10, height=8)


# Random Forests Error Plot
# Error rate of Random Forest
plot(rf1)

traindata <- as.data.frame(traindata)
# Tune mtry
t <- tuneRF(subset(train, select = -c(logbid1)), train[,c("logbid1")],
            stepFactor = 0.5,
            plot = TRUE,
            ntreeTry = 500,
            trace = TRUE,
            improve = 0.05)

################################################ The End ############################################

