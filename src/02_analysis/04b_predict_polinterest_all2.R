##################################################################################

library(tidyverse)
library(caret)
library(xgboost)
library(mboost)
library(pROC)
library(h2o)
library(SHAPforxgboost)
library(rtf)
library(gridExtra)
library(cowplot)

# Set path
# setwd("/home/r_uma_2019/respondi_eu/")

##################################################################################
# Setup
##################################################################################

# load data
load("./src/02_analysis/prep_predict_all.Rdata")

# Caret Setup

evalStats <- function(...) c(twoClassSummary(...),
                             defaultSummary(...),
                             mnLogLoss(...))

ctrl  <- trainControl(method = "cv",
                      number = 10,
                      summaryFunction = evalStats,
                      classProbs = TRUE,
                      verboseIter = TRUE)

# XGB Grid0

xgb_grid0 <- expand.grid(max_depth = c(1, 3, 5, 7, 9, 11),
                         nrounds = 100,
                         eta = 0.05,
                         min_child_weight = 0:5,
                         subsample = 1,
                         gamma = c(0, 0.5, 1),
                         colsample_bytree = 1)

mboost_grid <- expand.grid(mstop = c(100, 200, 300, 400, 500, 750),
                           prune = FALSE)

model_p1b <- paste("polinterest ~", paste(demo, collapse="+"), paste("+"), paste(media, collapse="+"))
model_p2b <- paste("polinterest ~", paste(pred_track_cat_dom_all, collapse="+"), 
                   paste("+"), paste(pred_track_cat_app_all, collapse="+"),
                   paste("+"), paste(pred_track_dom_all, collapse="+"), 
                   paste("+"), paste(pred_track_app_all, collapse="+"), paste("+ no_app_data"))
model_p3b <- paste("polinterest ~", paste(bert_pca_cluster_reduced_all, collapse="+"), paste("+ no_reduced_bert"))
model_p4b <- paste("polinterest ~", paste(topics_all, collapse="+"), paste("+ no_topics"))
model_p5b <- paste("polinterest ~", paste(demo, collapse="+"), paste("+"), paste(media, collapse="+"),
                   paste("+"), paste(pred_track_cat_dom_all, collapse="+"), 
                   paste("+"), paste(pred_track_cat_app_all, collapse="+"),
                   paste("+"), paste(pred_track_dom_all, collapse="+"), 
                   paste("+"), paste(pred_track_app_all, collapse="+"), paste("+ no_app_data"),
                   paste("+"), paste(bert_pca_cluster_reduced_all, collapse="+"), paste("+ no_reduced_bert"),
                   paste("+"), paste(topics_all, collapse="+"), paste("+ no_topics"))

h2o.init()
track_de_train_h2o <- as.h2o(track_de_train)
track_fr_train_h2o <- as.h2o(track_fr_train)
track_uk_train_h2o <- as.h2o(track_uk_train)
track_de_test_h2o <- as.h2o(track_de_test)
track_fr_test_h2o <- as.h2o(track_fr_test)
track_uk_test_h2o <- as.h2o(track_uk_test)

Y <- "polinterest"
X_p1b <- c(demo, media)
X_p2b <- c(pred_track_cat_dom_all, pred_track_cat_app_all, pred_track_dom_all, pred_track_app_all, "no_app_data")
X_p3b <- c(bert_pca_cluster_reduced_all, "no_reduced_bert")
X_p4b <- c(topics_all, "no_topics")

##################################################################################
# Models - XGBoost + mboost
##################################################################################

# Pol-Interest - Germany - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_de_p1ba <- train(",model_p1b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth1 <- xgb_de_p1ba$results$max_depth[which.min(xgb_de_p1ba$results[,"logLoss"])]
best_child1 <- xgb_de_p1ba$results$min_child_weight[which.min(xgb_de_p1ba$results[,"logLoss"])]
best_gamma1 <- xgb_de_p1ba$results$gamma[which.min(xgb_de_p1ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth1-1, best_depth1, best_depth1+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child1,
                        subsample = c(0.7, 1),
                        gamma = best_gamma1,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_p1b <- train(",model_p1b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_de_p1b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_p1b <- train(",model_p1b,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_de_p1b, metric = "ROC")

# Pol-Interest - Germany - categories + domains/apps

set.seed(48284)
eval(parse(text=paste("xgb_de_p2ba <- train(",model_p2b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth2 <- xgb_de_p2ba$results$max_depth[which.min(xgb_de_p2ba$results[,"logLoss"])]
best_child2 <- xgb_de_p2ba$results$min_child_weight[which.min(xgb_de_p2ba$results[,"logLoss"])]
best_gamma2 <- xgb_de_p2ba$results$gamma[which.min(xgb_de_p2ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth2-1, best_depth2, best_depth2+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child2,
                        subsample = c(0.7, 1),
                        gamma = best_gamma2,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_p2b <- train(",model_p2b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_de_p2b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_p2b <- train(",model_p2b,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_de_p2b, metric = "ROC")

# Pol-Interest - Germany - reduced BERT

set.seed(48284)
eval(parse(text=paste("xgb_de_p3ba <- train(",model_p3b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth3 <- xgb_de_p3ba$results$max_depth[which.min(xgb_de_p3ba$results[,"logLoss"])]
best_child3 <- xgb_de_p3ba$results$min_child_weight[which.min(xgb_de_p3ba$results[,"logLoss"])]
best_gamma3 <- xgb_de_p3ba$results$gamma[which.min(xgb_de_p3ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth3-1, best_depth3, best_depth3+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child3,
                        subsample = c(0.7, 1),
                        gamma = best_gamma3,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_p3b <- train(",model_p3b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_de_p3b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_p3b <- train(",model_p3b,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_de_p3b, metric = "ROC")

# Pol-Interest - Germany - topics

set.seed(48284)
eval(parse(text=paste("xgb_de_p4ba <- train(",model_p4b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth4 <- xgb_de_p4ba$results$max_depth[which.min(xgb_de_p4ba$results[,"logLoss"])]
best_child4 <- xgb_de_p4ba$results$min_child_weight[which.min(xgb_de_p4ba$results[,"logLoss"])]
best_gamma4 <- xgb_de_p4ba$results$gamma[which.min(xgb_de_p4ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth4-1, best_depth4, best_depth4+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child4,
                        subsample = c(0.7, 1),
                        gamma = best_gamma4,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_p4b <- train(",model_p4b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_de_p4b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_p4b <- train(",model_p4b,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_de_p4b, metric = "ROC")

# Pol-Interest - Germany - all

set.seed(48284)
eval(parse(text=paste("xgb_de_p5ba <- train(",model_p5b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth5 <- xgb_de_p5ba$results$max_depth[which.min(xgb_de_p5ba$results[,"logLoss"])]
best_child5 <- xgb_de_p5ba$results$min_child_weight[which.min(xgb_de_p5ba$results[,"logLoss"])]
best_gamma5 <- xgb_de_p5ba$results$gamma[which.min(xgb_de_p5ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth5-1, best_depth5, best_depth5+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child5,
                        subsample = c(0.7, 1),
                        gamma = best_gamma5,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_p5b <- train(",model_p5b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_de_p5b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_p5b <- train(",model_p5b,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_de_p5b, metric = "ROC")

# Pol-Interest - Germany - stacked model

xgb_de_p1c <- h2o.xgboost(
  x = X_p1b, y = Y, training_frame = track_de_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth1, learn_rate = 0.01, min_rows = best_child1, sample_rate = 0.7, min_split_improvement = best_gamma1, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_de_p2c <- h2o.xgboost(
  x = X_p2b, y = Y, training_frame = track_de_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth2, learn_rate = 0.01, min_rows = best_child2, sample_rate = 0.7, min_split_improvement = best_gamma2, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_de_p3c <- h2o.xgboost(
  x = X_p3b, y = Y, training_frame = track_de_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth3, learn_rate = 0.01, min_rows = best_child3, sample_rate = 0.7, min_split_improvement = best_gamma3, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_de_p4c <- h2o.xgboost(
  x = X_p4b, y = Y, training_frame = track_de_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth4, learn_rate = 0.01, min_rows = best_child4, sample_rate = 0.7, min_split_improvement = best_gamma4, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

glm_de_p6b <- h2o.stackedEnsemble(
  y = Y, training_frame = track_de_train_h2o,
  base_models = list(xgb_de_p1c, xgb_de_p2c, xgb_de_p3c, xgb_de_p4c),
  metalearner_algorithm = "glm", metalearner_nfolds = 10)

tab <- h2o.getModel(glm_de_p6b@model$metalearner$name)@model$coefficients_table
h2o.saveModel(glm_de_p6b)

rtffile <- RTF("p_de6_stacked.doc")
addTable(rtffile, cbind(tab[,1],round(tab[,2:3], digits = 3)))
done(rtffile)

##################################################################################

# Pol-Interest - France - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_fr_p1ba <- train(",model_p1b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth1 <- xgb_fr_p1ba$results$max_depth[which.min(xgb_fr_p1ba$results[,"logLoss"])]
best_child1 <- xgb_fr_p1ba$results$min_child_weight[which.min(xgb_fr_p1ba$results[,"logLoss"])]
best_gamma1 <- xgb_fr_p1ba$results$gamma[which.min(xgb_fr_p1ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth1-1, best_depth1, best_depth1+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child1,
                        subsample = c(0.7, 1),
                        gamma = best_gamma1,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_p1b <- train(",model_p1b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_fr_p1b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_p1b <- train(",model_p1b,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_fr_p1b, metric = "ROC")

# Pol-Interest - France - categories + domains/apps

set.seed(48284)
eval(parse(text=paste("xgb_fr_p2ba <- train(",model_p2b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth2 <- xgb_fr_p2ba$results$max_depth[which.min(xgb_fr_p2ba$results[,"logLoss"])]
best_child2 <- xgb_fr_p2ba$results$min_child_weight[which.min(xgb_fr_p2ba$results[,"logLoss"])]
best_gamma2 <- xgb_fr_p2ba$results$gamma[which.min(xgb_fr_p2ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth2-1, best_depth2, best_depth2+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child2,
                        subsample = c(0.7, 1),
                        gamma = best_gamma2,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_p2b <- train(",model_p2b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_fr_p2b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_p2b <- train(",model_p2b,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_fr_p2b, metric = "ROC")

# Pol-Interest - France - reduced BERT

set.seed(48284)
eval(parse(text=paste("xgb_fr_p3ba <- train(",model_p3b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth3 <- xgb_fr_p3ba$results$max_depth[which.min(xgb_fr_p3ba$results[,"logLoss"])]
best_child3 <- xgb_fr_p3ba$results$min_child_weight[which.min(xgb_fr_p3ba$results[,"logLoss"])]
best_gamma3 <- xgb_fr_p3ba$results$gamma[which.min(xgb_fr_p3ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth3-1, best_depth3, best_depth3+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child3,
                        subsample = c(0.7, 1),
                        gamma = best_gamma3,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_p3b <- train(",model_p3b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_fr_p3b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_p3b <- train(",model_p3b,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_fr_p3b, metric = "ROC")

# Pol-Interest - France - topics

set.seed(48284)
eval(parse(text=paste("xgb_fr_p4ba <- train(",model_p4b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth4 <- xgb_fr_p4ba$results$max_depth[which.min(xgb_fr_p4ba$results[,"logLoss"])]
best_child4 <- xgb_fr_p4ba$results$min_child_weight[which.min(xgb_fr_p4ba$results[,"logLoss"])]
best_gamma4 <- xgb_fr_p4ba$results$gamma[which.min(xgb_fr_p4ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth4-1, best_depth4, best_depth4+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child4,
                        subsample = c(0.7, 1),
                        gamma = best_gamma4,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_p4b <- train(",model_p4b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_fr_p4b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_p4b <- train(",model_p4b,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_fr_p4b, metric = "ROC")

# Pol-Interest - France - all

set.seed(48284)
eval(parse(text=paste("xgb_fr_p5ba <- train(",model_p5b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth5 <- xgb_fr_p5ba$results$max_depth[which.min(xgb_fr_p5ba$results[,"logLoss"])]
best_child5 <- xgb_fr_p5ba$results$min_child_weight[which.min(xgb_fr_p5ba$results[,"logLoss"])]
best_gamma5 <- xgb_fr_p5ba$results$gamma[which.min(xgb_fr_p5ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth5-1, best_depth5, best_depth5+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child5,
                        subsample = c(0.7, 1),
                        gamma = best_gamma5,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_p5b <- train(",model_p5b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_fr_p5b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_p5b <- train(",model_p5b,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_fr_p5b, metric = "ROC")

# Pol-Interest - France - stacked model

xgb_fr_p1c <- h2o.xgboost(
  x = X_p1b, y = Y, training_frame = track_fr_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth1, learn_rate = 0.01, min_rows = best_child1, sample_rate = 0.7, min_split_improvement = best_gamma1, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_fr_p2c <- h2o.xgboost(
  x = X_p2b, y = Y, training_frame = track_fr_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth2, learn_rate = 0.01, min_rows = best_child2, sample_rate = 0.7, min_split_improvement = best_gamma2, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_fr_p3c <- h2o.xgboost(
  x = X_p3b, y = Y, training_frame = track_fr_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth3, learn_rate = 0.01, min_rows = best_child3, sample_rate = 0.7, min_split_improvement = best_gamma3, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_fr_p4c <- h2o.xgboost(
  x = X_p4b, y = Y, training_frame = track_fr_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth4, learn_rate = 0.01, min_rows = best_child4, sample_rate = 0.7, min_split_improvement = best_gamma4, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

glm_fr_p6b <- h2o.stackedEnsemble(
  y = Y, training_frame = track_fr_train_h2o,
  base_models = list(xgb_fr_p1c, xgb_fr_p2c, xgb_fr_p3c, xgb_fr_p4c),
  metalearner_algorithm = "glm", metalearner_nfolds = 10)

tab <- h2o.getModel(glm_fr_p6b@model$metalearner$name)@model$coefficients_table
h2o.saveModel(glm_fr_p6b)

rtffile <- RTF("p_fr6_stacked.doc")
addTable(rtffile, cbind(tab[,1],round(tab[,2:3], digits = 3)))
done(rtffile)

##################################################################################

# Pol-Interest - UK - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_uk_p1ba <- train(",model_p1b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth1 <- xgb_uk_p1ba$results$max_depth[which.min(xgb_uk_p1ba$results[,"logLoss"])]
best_child1 <- xgb_uk_p1ba$results$min_child_weight[which.min(xgb_uk_p1ba$results[,"logLoss"])]
best_gamma1 <- xgb_uk_p1ba$results$gamma[which.min(xgb_uk_p1ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth1-1, best_depth1, best_depth1+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child1,
                        subsample = c(0.7, 1),
                        gamma = best_gamma1,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_p1b <- train(",model_p1b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_uk_p1b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_p1b <- train(",model_p1b,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_uk_p1b, metric = "ROC")

# Pol-Interest - UK - categories + domains/apps

set.seed(48284)
eval(parse(text=paste("xgb_uk_p2ba <- train(",model_p2b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth2 <- xgb_uk_p2ba$results$max_depth[which.min(xgb_uk_p2ba$results[,"logLoss"])]
best_child2 <- xgb_uk_p2ba$results$min_child_weight[which.min(xgb_uk_p2ba$results[,"logLoss"])]
best_gamma2 <- xgb_uk_p2ba$results$gamma[which.min(xgb_uk_p2ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth2-1, best_depth2, best_depth2+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child2,
                        subsample = c(0.7, 1),
                        gamma = best_gamma2,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_p2b <- train(",model_p2b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_uk_p2b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_p2b <- train(",model_p2b,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_uk_p2b, metric = "ROC")

# Pol-Interest - UK - reduced BERT

set.seed(48284)
eval(parse(text=paste("xgb_uk_p3ba <- train(",model_p3b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth3 <- xgb_uk_p3ba$results$max_depth[which.min(xgb_uk_p3ba$results[,"logLoss"])]
best_child3 <- xgb_uk_p3ba$results$min_child_weight[which.min(xgb_uk_p3ba$results[,"logLoss"])]
best_gamma3 <- xgb_uk_p3ba$results$gamma[which.min(xgb_uk_p3ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth3-1, best_depth3, best_depth3+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child3,
                        subsample = c(0.7, 1),
                        gamma = best_gamma3,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_p3b <- train(",model_p3b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_uk_p3b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_p3b <- train(",model_p3b,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_uk_p3b, metric = "ROC")

# Pol-Interest - UK - topics

set.seed(48284)
eval(parse(text=paste("xgb_uk_p4ba <- train(",model_p4b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth4 <- xgb_uk_p4ba$results$max_depth[which.min(xgb_uk_p4ba$results[,"logLoss"])]
best_child4 <- xgb_uk_p4ba$results$min_child_weight[which.min(xgb_uk_p4ba$results[,"logLoss"])]
best_gamma4 <- xgb_uk_p4ba$results$gamma[which.min(xgb_uk_p4ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth4-1, best_depth4, best_depth4+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child4,
                        subsample = c(0.7, 1),
                        gamma = best_gamma4,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_p4b <- train(",model_p4b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_uk_p4b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_p4b <- train(",model_p4b,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_uk_p4b, metric = "ROC")

# Pol-Interest - UK - all

set.seed(48284)
eval(parse(text=paste("xgb_uk_p5ba <- train(",model_p5b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth5 <- xgb_uk_p5ba$results$max_depth[which.min(xgb_uk_p5ba$results[,"logLoss"])]
best_child5 <- xgb_uk_p5ba$results$min_child_weight[which.min(xgb_uk_p5ba$results[,"logLoss"])]
best_gamma5 <- xgb_uk_p5ba$results$gamma[which.min(xgb_uk_p5ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth5-1, best_depth5, best_depth5+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child5,
                        subsample = c(0.7, 1),
                        gamma = best_gamma5,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_p5b <- train(",model_p5b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_uk_p5b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_p5b <- train(",model_p5b,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_uk_p5b, metric = "ROC")

# Pol-Interest - UK - stacked model

xgb_uk_p1c <- h2o.xgboost(
  x = X_p1b, y = Y, training_frame = track_uk_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth1, learn_rate = 0.01, min_rows = best_child1, sample_rate = 0.7, min_split_improvement = best_gamma1, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_uk_p2c <- h2o.xgboost(
  x = X_p2b, y = Y, training_frame = track_uk_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth2, learn_rate = 0.01, min_rows = best_child2, sample_rate = 0.7, min_split_improvement = best_gamma2, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_uk_p3c <- h2o.xgboost(
  x = X_p3b, y = Y, training_frame = track_uk_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth3, learn_rate = 0.01, min_rows = best_child3, sample_rate = 0.7, min_split_improvement = best_gamma3, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_uk_p4c <- h2o.xgboost(
  x = X_p4b, y = Y, training_frame = track_uk_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth4, learn_rate = 0.01, min_rows = best_child4, sample_rate = 0.7, min_split_improvement = best_gamma4, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

glm_uk_p6b <- h2o.stackedEnsemble(
  y = Y, training_frame = track_uk_train_h2o,
  base_models = list(xgb_uk_p1c, xgb_uk_p2c, xgb_uk_p3c, xgb_uk_p4c),
  metalearner_algorithm = "glm", metalearner_nfolds = 10)

tab <- h2o.getModel(glm_uk_p6b@model$metalearner$name)@model$coefficients_table
h2o.saveModel(glm_uk_p6b)

rtffile <- RTF("p_uk6_stacked.doc")
addTable(rtffile, cbind(tab[,1],round(tab[,2:3], digits = 3)))
done(rtffile)

##################################################################################
# Variable Importance
##################################################################################

plot(varImp(xgb_de_p1b), top = 10)
plot(varImp(xgb_de_p2b), top = 10)
plot(varImp(xgb_de_p3b), top = 10)
plot(varImp(xgb_de_p4b), top = 10)
plot(varImp(xgb_de_p5b), top = 10)

plot(varImp(xgb_fr_p1b), top = 10)
plot(varImp(xgb_fr_p2b), top = 10)
plot(varImp(xgb_fr_p3b), top = 10)
plot(varImp(xgb_fr_p4b), top = 10)
plot(varImp(xgb_fr_p5b), top = 10)

plot(varImp(xgb_uk_p1b), top = 10)
plot(varImp(xgb_uk_p2b), top = 10)
plot(varImp(xgb_uk_p3b), top = 10)
plot(varImp(xgb_uk_p4b), top = 10)
plot(varImp(xgb_uk_p5b), top = 10)

##################################################################################
# SHAP
##################################################################################

# Pol-Interest - Germany - socio_demo + reduced BERT
xgb_de_5 <- xgb_de_p5b$finalModel
eval(parse(text=paste("x_data <- model.matrix(",model_p5b," , track_de_train)[, -1]")))
shap_long <- shap.prep(xgb_de_5, X_train = x_data, top_n = 20)
shap.plot.summary(shap_long, dilute = T)

# Pol-Interest - France - socio_demo + reduced BERT
xgb_fr_5 <- xgb_fr_p5b$finalModel
eval(parse(text=paste("x_data <- model.matrix(",model_p5b," , track_fr_train)[, -1]")))
shap_long <- shap.prep(xgb_fr_5, X_train = x_data, top_n = 20)
shap.plot.summary(shap_long, dilute = T)

# Pol-Interest - UK - socio_demo + reduced BERT
xgb_uk_5 <- xgb_uk_p5b$finalModel
eval(parse(text=paste("x_data <- model.matrix(",model_p5b," , track_uk_train)[, -1]")))
shap_long <- shap.prep(xgb_uk_5, X_train = x_data, top_n = 20)
shap.plot.summary(shap_long, dilute = T)

##################################################################################
# Compare CV performance
##################################################################################

resamps_de <- resamples(list(xgb_de_p1b, xgb_de_p2b, xgb_de_p3b,
                             xgb_de_p4b, xgb_de_p5b))

resamps_fr <- resamples(list(xgb_fr_p1b, xgb_fr_p2b, xgb_fr_p3b,
                             xgb_fr_p4b, xgb_fr_p5b))

resamps_uk <- resamples(list(xgb_uk_p1b, xgb_uk_p2b, xgb_uk_p3b,
                             xgb_uk_p4b, xgb_uk_p5b))

sum_resamps_de <- summary(resamps_de)
sum_resamps_fr <- summary(resamps_fr)
sum_resamps_uk <- summary(resamps_uk)

polinterest_cv_de <- data.frame("Country" = "Germany", 
                                "Model" = names(sum_resamps_de$statistics$ROC[, 4]),
                                "AUC" = sum_resamps_de$statistics$ROC[, 4])
polinterest_cv_de <- polinterest_cv_de %>% add_row(Country = "Germany", Model = "Model6", AUC = h2o.auc(glm_de_p6b, xval = T))

polinterest_cv_fr <- data.frame("Country" = "France", 
                                "Model" = names(sum_resamps_fr$statistics$ROC[, 4]),
                                "AUC" = sum_resamps_fr$statistics$ROC[, 4])
polinterest_cv_fr <- polinterest_cv_fr %>% add_row(Country = "France", Model = "Model6", AUC = h2o.auc(glm_fr_p6b, xval = T))

polinterest_cv_uk <- data.frame("Country" = "UK", 
                                "Model" = names(sum_resamps_uk$statistics$ROC[, 4]),
                                "AUC" = sum_resamps_uk$statistics$ROC[, 4])
polinterest_cv_uk <- polinterest_cv_uk %>% add_row(Country = "UK", Model = "Model6", AUC = h2o.auc(glm_uk_p6b, xval = T))

polinterest_cv_roc <- as.data.frame(rbind(polinterest_cv_de, 
                                          polinterest_cv_fr, 
                                          polinterest_cv_uk))

polinterest_cv_roc$Model <- fct_recode(polinterest_cv_roc$Model,
                                 "Model1: Demo" = "Model1",
                                 "Model2: Categories" = "Model2",
                                 "Model3: BERT" = "Model3",
                                 "Model4: Topics" = "Model4",
                                 "Model5: All" = "Model5",
                                 "Model6: Stacking" = "Model6")

##################################################################################
# Predict in test data
##################################################################################

p_xgb_de_p1b <- predict(xgb_de_p1b, newdata = track_de_test, type = "prob")
p_xgb_de_p2b <- predict(xgb_de_p2b, newdata = track_de_test, type = "prob")
p_xgb_de_p3b <- predict(xgb_de_p3b, newdata = track_de_test, type = "prob")
p_xgb_de_p4b <- predict(xgb_de_p4b, newdata = track_de_test, type = "prob")
p_xgb_de_p5b <- predict(xgb_de_p5b, newdata = track_de_test, type = "prob")
p_glm_de_p6b <- h2o.predict(glm_de_p6b, newdata = track_de_test_h2o)

p_xgb_fr_p1b <- predict(xgb_fr_p1b, newdata = track_fr_test, type = "prob")
p_xgb_fr_p2b <- predict(xgb_fr_p2b, newdata = track_fr_test, type = "prob")
p_xgb_fr_p3b <- predict(xgb_fr_p3b, newdata = track_fr_test, type = "prob")
p_xgb_fr_p4b <- predict(xgb_fr_p4b, newdata = track_fr_test, type = "prob")
p_xgb_fr_p5b <- predict(xgb_fr_p5b, newdata = track_fr_test, type = "prob")
p_glm_fr_p6b <- h2o.predict(glm_fr_p6b, newdata = track_fr_test_h2o)

p_xgb_uk_p1b <- predict(xgb_uk_p1b, newdata = track_uk_test, type = "prob")
p_xgb_uk_p2b <- predict(xgb_uk_p2b, newdata = track_uk_test, type = "prob")
p_xgb_uk_p3b <- predict(xgb_uk_p3b, newdata = track_uk_test, type = "prob")
p_xgb_uk_p4b <- predict(xgb_uk_p4b, newdata = track_uk_test, type = "prob")
p_xgb_uk_p5b <- predict(xgb_uk_p5b, newdata = track_uk_test, type = "prob")
p_glm_uk_p6b <- h2o.predict(glm_uk_p6b, newdata = track_uk_test_h2o)

roc_xgb_de_p1b <- roc(response = track_de_test$polinterest, predictor = p_xgb_de_p1b$low, levels = c("low", "high"), direction = ">")$auc
roc_xgb_de_p2b <- roc(response = track_de_test$polinterest, predictor = p_xgb_de_p2b$low, levels = c("low", "high"), direction = ">")$auc
roc_xgb_de_p3b <- roc(response = track_de_test$polinterest, predictor = p_xgb_de_p3b$low, levels = c("low", "high"), direction = ">")$auc
roc_xgb_de_p4b <- roc(response = track_de_test$polinterest, predictor = p_xgb_de_p4b$low, levels = c("low", "high"), direction = ">")$auc
roc_xgb_de_p5b <- roc(response = track_de_test$polinterest, predictor = p_xgb_de_p5b$low, levels = c("low", "high"), direction = ">")$auc
roc_glm_de_p6b <- roc(response = track_de_test$polinterest, predictor = as.vector(p_glm_de_p6b$low))$auc

polinterest_test_de <- data.frame("Country" = "Germany", 
                                  "Model" = c("Model1: Demo", "Model2: Categories", "Model3: BERT", "Model4: Topics", "Model5: All", "Model6: Stacking"),
                                  "AUC" = rbind(roc_xgb_de_p1b, roc_xgb_de_p2b, roc_xgb_de_p3b, roc_xgb_de_p4b, roc_xgb_de_p5b, roc_glm_de_p6b))

roc_xgb_fr_p1b <- roc(response = track_fr_test$polinterest, predictor = p_xgb_fr_p1b$low, levels = c("low", "high"), direction = ">")$auc
roc_xgb_fr_p2b <- roc(response = track_fr_test$polinterest, predictor = p_xgb_fr_p2b$low, levels = c("low", "high"), direction = ">")$auc
roc_xgb_fr_p3b <- roc(response = track_fr_test$polinterest, predictor = p_xgb_fr_p3b$low, levels = c("low", "high"), direction = ">")$auc
roc_xgb_fr_p4b <- roc(response = track_fr_test$polinterest, predictor = p_xgb_fr_p4b$low, levels = c("low", "high"), direction = ">")$auc
roc_xgb_fr_p5b <- roc(response = track_fr_test$polinterest, predictor = p_xgb_fr_p5b$low, levels = c("low", "high"), direction = ">")$auc
roc_glm_fr_p6b <- roc(response = track_fr_test$polinterest, predictor = as.vector(p_glm_fr_p6b$low))$auc

polinterest_test_fr <- data.frame("Country" = "France", 
                                  "Model" = c("Model1: Demo", "Model2: Categories", "Model3: BERT", "Model4: Topics", "Model5: All", "Model6: Stacking"),
                                  "AUC" = rbind(roc_xgb_fr_p1b, roc_xgb_fr_p2b, roc_xgb_fr_p3b, roc_xgb_fr_p4b, roc_xgb_fr_p5b, roc_glm_fr_p6b))

roc_xgb_uk_p1b <- roc(response = track_uk_test$polinterest, predictor = p_xgb_uk_p1b$low, levels = c("low", "high"), direction = ">")$auc
roc_xgb_uk_p2b <- roc(response = track_uk_test$polinterest, predictor = p_xgb_uk_p2b$low, levels = c("low", "high"), direction = ">")$auc
roc_xgb_uk_p3b <- roc(response = track_uk_test$polinterest, predictor = p_xgb_uk_p3b$low, levels = c("low", "high"), direction = ">")$auc
roc_xgb_uk_p4b <- roc(response = track_uk_test$polinterest, predictor = p_xgb_uk_p4b$low, levels = c("low", "high"), direction = ">")$auc
roc_xgb_uk_p5b <- roc(response = track_uk_test$polinterest, predictor = p_xgb_uk_p5b$low, levels = c("low", "high"), direction = ">")$auc
roc_glm_uk_p6b <- roc(response = track_uk_test$polinterest, predictor = as.vector(p_glm_uk_p6b$low))$auc

polinterest_test_uk <- data.frame("Country" = "UK", 
                                  "Model" = c("Model1: Demo", "Model2: Categories", "Model3: BERT", "Model4: Topics", "Model5: All", "Model6: Stacking"),
                                  "AUC" = rbind(roc_xgb_uk_p1b, roc_xgb_uk_p2b, roc_xgb_uk_p3b, roc_xgb_uk_p4b, roc_xgb_uk_p5b, roc_glm_uk_p6b))

polinterest_test_roc <- as.data.frame(rbind(polinterest_test_de, 
                                            polinterest_test_fr, 
                                            polinterest_test_uk))

# Combine and plot results

polinterest_roc <- bind_rows("CV" = polinterest_cv_roc, "Test" = polinterest_test_roc, .id = "type")

ggplot(polinterest_roc, aes(Model, Country)) + 
  geom_tile(aes(fill = AUC), colour = "white") + 
  geom_text(aes(label = round(AUC, 2)), size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  facet_grid(cols = vars(type)) +
  labs(x = "", y = "") +
  scale_y_discrete(limits = rev(levels(polinterest_roc$Country))) +
  theme(legend.position = "none",
        text = element_text(size = 15),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

p1 <- ggplot(polinterest_cv_roc, aes(Model, Country)) + 
  geom_tile(aes(fill = AUC), colour = "white") + 
  geom_text(aes(label = round(AUC, 2)), size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(x = "", y = "") +
  ggtitle("CV in Training Set") +
  scale_y_discrete(limits = c("UK", "France", "Germany"), labels = c("DE, FR, 50% UK", "DE, UK, 50% FR", "FR, UK, 50% DE")) +
  theme(legend.position = "none",
        text = element_text(size = 15),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

p2 <- ggplot(polinterest_test_roc, aes(Model, Country)) + 
  geom_tile(aes(fill = AUC), colour = "white") + 
  geom_text(aes(label = round(AUC, 2)), size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(x = "", y = "") +
  ggtitle("Test Set") +
  scale_y_discrete(limits = c("UK", "France", "Germany"), labels = c("50% UK", "50% FR", "50% DE")) +
  theme(legend.position = "none",
        text = element_text(size = 15),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

g <- plot_grid(p1, p2, align = "h", ncol = 2, rel_widths = c(1.15, 1))

ggsave("p_roc.png", g, width = 9, height = 6.5)

# save.image("p_results_all2.RDATA")
