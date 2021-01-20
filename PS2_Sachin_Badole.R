title: "Problem Set 2"
author: "Sachin Badole"
date: "10/08/2020"

#clear workspace
rm(list=ls())

#setting up working directory
setwd("C:/Users/Sachin/Desktop/725/Problem set 2")

#load in packages
library(data.table)
library(glmnet)
require("data.table") 
require("glmnet")

#############################################################################
# Generate functions
#############################################################################
merge_fn = function(D,d,names,indicator=T) {
  colnames(d)[1] = names[1]
  D = merge(D,d,by=names[1])
  
  colnames(d)[1] = names[2]
  D = merge(D,d,by=c(names[2]))
  
  if (indicator) {#at least one must be 1
    D[,names[3]] = ((D[,(ncol(D)-1)] + D[,ncol(D)])>0)*1
  } else { #geometric mean
    D[,names[3]] = sqrt(D[,(ncol(D)-1)]*D[,ncol(D)])
  }
  
  return(D[,-c(ncol(D)-1,ncol(D)-2)])
}
#############################################################################
generate_indices = function(n,per) {
  return(sample.int(n,round(per*n)))
}
#############################################################################
colinear_terms = function(y,covariates) {
  model = lm(data.frame(y,covariates))
  return(names(model$coefficients[which(is.na(model$coefficients))]))
}
#############################################################################
linear_model = function(data,noise,test_ind) {
  
  if (!noise) data = data[,1:7]
  
  model = lm(data[-test_ind,])
  mse = mean((predict(model,data[test_ind,2:ncol(data)]) - data$num_carriers[test_ind])^2)
  
  return(mse)
} 
#############################################################################
linear_model_cross_terms = function(data,noise,test_ind) {
  y = data$num_carriers
  if (noise) {
    covariates = data[,2:ncol(data)]
  } else {covariates =  data[,2:7]}
  
  covariates = data.frame(poly(as.matrix(covariates),degree=order,raw=T))
  
  model = lm(data.frame(y,covariates)[-test_ind,])
  
  mse = mean((predict(model,covariates[test_ind,]) - y[test_ind])^2)
  
  return(mse)
}
#############################################################################
covariate_selection = function(data,noise,test_ind,cross_terms) {
  y = data$num_carriers
  
  if (noise) {
    covariates = data[,2:ncol(data)]
  } else {covariates = data[,2:7]}
  
  if (cross_terms) {
    covariates = data.frame(poly(as.matrix(covariates),degree=order,raw=T))
    covariates = covariates[,-which(colnames(covariates)%in%colinear_terms(y[-test_ind],covariates[-test_ind,]))]
  }
  covariates_save = covariates
  omit = NULL
  M = matrix(nrow=ncol(covariates)-1,ncol=3)
  outer_iters = ncol(covariates) - 1
  for (outer in 1:outer_iters) {
    criterion = numeric(ncol(covariates))
    for (k in ncol(covariates):1) {
      fit = lm(data.frame(y[-test_ind],covariates[-test_ind,-k]))
      criterion[k] = summary(fit)$adj.r.squared #rsquared
      #criterion[k] = BIC(fit) #BIC
    }
    remove = which(criterion==max(criterion))[1]
    omit = c(omit,colnames(covariates)[remove])
    covariates = covariates[,-remove]
    
    
    M[outer,] = c(outer,which(criterion==max(criterion))[1],
                  criterion[which(criterion==max(criterion))[1]])
  }
  
  minimizer = which(M[,3]==max(M[,3]))[length(which(M[,3]==max(M[,3])))]
  
  removed_terms = omit[1:(minimizer)]
  
  model = lm(data.frame(y[-test_ind],
                        covariates_save[-test_ind,][,!colnames(covariates_save[-test_ind,])%in%removed_terms]))
  
  mse = mean((predict(model,covariates_save[test_ind,]) - y[test_ind])^2)
  
  return(mse)
}
#############################################################################
cv = function(lambda,data,noise,test_indices,cross_terms,ridge) {
  y = data$num_carriers
  if (noise) {
    covariates = data[,2:ncol(data)]
  } else {covariates = data[,2:7]}
  
  if (cross_terms) {
    covariates = data.frame(poly(as.matrix(covariates),degree=order,raw=T))
    covariates = covariates[,-which(colnames(covariates)%in%colinear_terms(y[-test_ind],covariates[-test_ind,]))]
  }
  
  train = data.frame(y[-test_ind],covariates[-test_ind,])
  
  sum = 0
  for (k in 1:10) {
    
    test_indices_cv = seq(floor(seq(1,nrow(train),length.out=11))[k],
                          floor(seq(1,nrow(train),length.out=11))[k+1],1)
    
    test_x = as.matrix(train[test_indices_cv,2:ncol(train)])
    train_x = as.matrix(train[-test_indices_cv,2:ncol(train)])
    
    test_y = train[,1][test_indices_cv]
    train_y = train[,1][-test_indices_cv]
    
    if (ridge) {
      fit = glmnet(train_x,train_y,alpha=0,lambda=lambda)
      sum = sum + sum((predict(fit,test_x,s=lambda) - test_y)^2)
    } else {
      fit = glmnet(train_x,train_y,alpha=1,lambda=lambda)
      sum = sum + sum((predict(fit,test_x,s=lambda) - test_y)^2)
    }
  }
  return(sum)
}
#############################################################################
ridge_lasso = function(data,noise,test_ind,lambda,cross_terms,ridge) {
  y = data$num_carriers
  
  if (noise) {
    covariates = data[,2:ncol(data)]
  } else {covariates = data[,2:7]}
  
  if (cross_terms) {
    covariates = data.frame(poly(as.matrix(covariates),degree=order,raw=T))
    covariates = covariates[,-which(colnames(covariates)%in%colinear_terms(data$num_carriers[-test_ind],covariates[-test_ind,]))]
  }
  
  if (ridge) {
    alpha = 0
  } else {
    alpha = 1
  }
  
  fit = glmnet(as.matrix(covariates[-test_ind,]),y[-test_ind],alpha=alpha,lambda=lambda)
  mse = mean((predict(fit,as.matrix(covariates[test_ind,]),s=lambda) - y[test_ind])^2)
  
  return(mse)
}
#############################################################################
# Solution for Problem set 2

#print results?
print_results=T

#order of polynomials
order = 2

#set working directory
#setwd("~/Econ_690")

source("ps_2_functions.R")

load(file="airline_data_market_level.R")

#Get list of Hub airports and airline code table
load(file="lookup_and_hub_r.R")

#get vacation routes
load(file="vacations.R")

#get MSA income data
load(file="data_income.R")

#get list of slot_controlled airports
load(file="slot_controlled.R")

#no scientific notation
options(scipen=999)

#no warnings from colinearity in linear models
options(warn=-1)
##########################################################################################
#hubs
hubs = data.frame(lookup_and_hub$Code,rowSums(lookup_and_hub[,4:ncol(lookup_and_hub)]))

datam = merge_fn(datam,hubs,names=c("origin_airport_id","dest_airport_id","hub_route"))


#vacation routes
datam = merge_fn(datam,vacations,names=c("origin_city","dest_city","vacation_route"))

#slot controlled routes
datam = merge_fn(datam,slot_controlled,names=c("origin_airport_id","dest_airport_id","slot_controlled"))

#income
datam = merge_fn(datam,msa_income,names=c("origin_city","dest_city","market_income"),indicator=F)
##########################################################
datam = datam[order(datam$origin_airport_id,datam$dest_airport_id),]
##########################################################
set.seed(0)
datam$noise1 = datam$average_distance_m + rnorm(nrow(datam),0,1)
datam$noise2 = datam$market_size + rnorm(nrow(datam),0,1)
datam$noise3 = datam$market_income + rnorm(nrow(datam),0,1)

datam = datam[,c("num_carriers","market_income","market_size",
                          "slot_controlled","vacation_route","hub_route","average_distance_m",
                          "noise1","noise2","noise3")]

#per = .98; noise = T
for (per in c(.5,.98)) {
  #test markets
  set.seed(0)
  test_ind = generate_indices(nrow(datam),per)
  for (noise in c(F,T)) {

    ##########################################################
    #linear model
    linear_mse = linear_model(datam,noise,test_ind); #linear_mse
    ##########################################################
    #linear model cross terms
    full_mse = linear_model_cross_terms(datam,noise,test_ind); #full_mse
    ##########################################################
    #covariate selection
    cs_mse = covariate_selection(datam,noise,test_ind,cross_terms=T); #cs_mse
    ##########################################################
    #Lasso and Ridge lambda selection through cv
    ridge_lambda = optimize(cv,interval=c(0,10),data=datam,noise=noise,test_indices=test_ind,cross_terms=T,ridge=T)$minimum
    lasso_lambda = optimize(cv,interval=c(0,1),data=datam,noise=noise,test_indices=test_ind,cross_terms=T,ridge=F)$minimum
    ##########################################################
    #Lasso and Ridge
    ridge_mse = ridge_lasso(datam,noise=noise,test_ind=test_ind,lambda=ridge_lambda,cross_terms=T,ridge=T); #ridge_mse
    lasso_mse = ridge_lasso(datam,noise=noise,test_ind=test_ind,lambda=lasso_lambda,cross_terms=T,ridge=F); #lasso_mse
    
    if (print_results) {
      print("##########################################################")
      print(paste(per,"percent, noise is ", noise,sep = " "))
      print(paste("linear_mse is ",linear_mse))
      print(paste("full_mse is ",full_mse))
      print(paste("covariate_shrink_mse is ",cs_mse))
      print(paste("ridge_mse is ",ridge_mse))
      print(paste("lasso_mse is ",lasso_mse))
      print("##########################################################")
    }
  }
}


# Results
# [1] "##########################################################"
# [1] "0.5 percent, noise is  FALSE"
# [1] "linear_mse is  2.20838296454737"
# [1] "full_mse is  1.90103858369537"
# [1] "covariate_shrink_mse is  1.90193665788821"
# [1] "ridge_mse is  1.90834672687366"
# [1] "lasso_mse is  1.92525893751575"
# [1] "##########################################################"
# [1] "##########################################################"
# [1] "0.5 percent, noise is  TRUE"
# [1] "linear_mse is  2.20773707368945"
# [1] "full_mse is  1.91963842578358"
# [1] "covariate_shrink_mse is  1.91429196098368"
# [1] "ridge_mse is  1.90652446387727"
# [1] "lasso_mse is  1.92499315698279"
# [1] "##########################################################"
# [1] "##########################################################"
# [1] "0.98 percent, noise is  FALSE"
# [1] "linear_mse is  2.47100062745029"
# [1] "full_mse is  3.2780548491551"
# [1] "covariate_shrink_mse is  2.99904659046479"
# [1] "ridge_mse is  2.3439068820102"
# [1] "lasso_mse is  2.36359246298439"
# [1] "##########################################################"
# [1] "##########################################################"
# [1] "0.98 percent, noise is  TRUE"
# [1] "linear_mse is  2.64060680756035"
# [1] "full_mse is  4.56920828546955"
# [1] "covariate_shrink_mse is  3.77507356961129"
# [1] "ridge_mse is  2.33854862633867"
# [1] "lasso_mse is  2.36345243421244"
# [1] "##########################################################"
