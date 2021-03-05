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

model_c1b <- paste("changed ~", paste(demo, collapse="+"), paste("+"), paste(media, collapse="+"))
model_c2b <- paste("changed ~", paste(pred_track_cat_dom_all, collapse="+"), 
                   paste("+"), paste(pred_track_cat_app_all, collapse="+"),
                   paste("+"), paste(pred_track_dom_all, collapse="+"), 
                   paste("+"), paste(pred_track_app_all, collapse="+"), paste("+ no_app_data"))
model_c3b <- paste("changed ~", paste(bert_pca_cluster_reduced_all, collapse="+"), paste("+ no_reduced_bert"))
model_c4b <- paste("changed ~", paste(topics_all, collapse="+"), paste("+ no_topics"))
model_c5b <- paste("changed ~", paste(demo, collapse="+"), paste("+"), paste(media, collapse="+"),
                   paste("+"), paste(pred_track_cat_dom_all, collapse="+"), 
                   paste("+"), paste(pred_track_cat_app_all, collapse="+"),
                   paste("+"), paste(pred_track_dom_all, collapse="+"), 
                   paste("+"), paste(pred_track_app_all, collapse="+"), paste("+ no_app_data"),
                   paste("+"), paste(bert_pca_cluster_reduced_all, collapse="+"), paste("+ no_reduced_bert"),
                   paste("+"), paste(topics_all, collapse="+"), paste("+ no_topics"))

h2o.init()
track_de_train_h2o <- as.h2o(drop_na(track_de_train, changed))
track_fr_train_h2o <- as.h2o(drop_na(track_fr_train, changed))
track_uk_train_h2o <- as.h2o(drop_na(track_uk_train, changed))
track_de_test_h2o <- as.h2o(track_de_test)
track_fr_test_h2o <- as.h2o(track_fr_test)
track_uk_test_h2o <- as.h2o(track_uk_test)

Y <- "changed"
X_c1b <- c(demo, media)
X_c2b <- c(pred_track_cat_dom_all, pred_track_cat_app_all, pred_track_dom_all, pred_track_app_all, "no_app_data")
X_c3b <- c(bert_pca_cluster_reduced_all, "no_reduced_bert")
X_c4b <- c(topics_all, "no_topics")

##################################################################################
# Models - XGBoost + mboost
##################################################################################

# Changed - Germany - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_de_c1ba <- train(",model_c1b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth1 <- xgb_de_c1ba$results$max_depth[which.min(xgb_de_c1ba$results[,"logLoss"])]
best_child1 <- xgb_de_c1ba$results$min_child_weight[which.min(xgb_de_c1ba$results[,"logLoss"])]
best_gamma1 <- xgb_de_c1ba$results$gamma[which.min(xgb_de_c1ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth1-1, best_depth1, best_depth1+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child1,
                        subsample = c(0.7, 1),
                        gamma = best_gamma1,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_c1b <- train(",model_c1b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_de_c1b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_c1b <- train(",model_c1b,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_de_c1b, metric = "ROC")

# Changed - Germany - categories + domains/apps

set.seed(48284)
eval(parse(text=paste("xgb_de_c2ba <- train(",model_c2b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth2 <- xgb_de_c2ba$results$max_depth[which.min(xgb_de_c2ba$results[,"logLoss"])]
best_child2 <- xgb_de_c2ba$results$min_child_weight[which.min(xgb_de_c2ba$results[,"logLoss"])]
best_gamma2 <- xgb_de_c2ba$results$gamma[which.min(xgb_de_c2ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth2-1, best_depth2, best_depth2+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child2,
                        subsample = c(0.7, 1),
                        gamma = best_gamma2,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_c2b <- train(",model_c2b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_de_c2b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_c2b <- train(",model_c2b,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_de_c2b, metric = "ROC")

# Changed - Germany - reduced BERT

set.seed(48284)
eval(parse(text=paste("xgb_de_c3ba <- train(",model_c3b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth3 <- xgb_de_c3ba$results$max_depth[which.min(xgb_de_c3ba$results[,"logLoss"])]
best_child3 <- xgb_de_c3ba$results$min_child_weight[which.min(xgb_de_c3ba$results[,"logLoss"])]
best_gamma3 <- xgb_de_c3ba$results$gamma[which.min(xgb_de_c3ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth3-1, best_depth3, best_depth3+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child3,
                        subsample = c(0.7, 1),
                        gamma = best_gamma3,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_c3b <- train(",model_c3b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_de_c3b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_c3b <- train(",model_c3b,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_de_c3b, metric = "ROC")

# Changed - Germany - topics

set.seed(48284)
eval(parse(text=paste("xgb_de_c4ba <- train(",model_c4b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth4 <- xgb_de_c4ba$results$max_depth[which.min(xgb_de_c4ba$results[,"logLoss"])]
best_child4 <- xgb_de_c4ba$results$min_child_weight[which.min(xgb_de_c4ba$results[,"logLoss"])]
best_gamma4 <- xgb_de_c4ba$results$gamma[which.min(xgb_de_c4ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth4-1, best_depth4, best_depth4+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child4,
                        subsample = c(0.7, 1),
                        gamma = best_gamma4,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_c4b <- train(",model_c4b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_de_c4b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_c4b <- train(",model_c4b,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_de_c4b, metric = "ROC")

# Changed - Germany - all

set.seed(48284)
eval(parse(text=paste("xgb_de_c5ba <- train(",model_c5b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth5 <- xgb_de_c5ba$results$max_depth[which.min(xgb_de_c5ba$results[,"logLoss"])]
best_child5 <- xgb_de_c5ba$results$min_child_weight[which.min(xgb_de_c5ba$results[,"logLoss"])]
best_gamma5 <- xgb_de_c5ba$results$gamma[which.min(xgb_de_c5ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth5-1, best_depth5, best_depth5+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child5,
                        subsample = c(0.7, 1),
                        gamma = best_gamma5,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_c5b <- train(",model_c5b,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_de_c5b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_c5b <- train(",model_c5b,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_de_c5b, metric = "ROC")

# Changed - Germany - stacked model

xgb_de_c1c <- h2o.xgboost(
  x = X_c1b, y = Y, training_frame = track_de_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth1, learn_rate = 0.01, min_rows = best_child1, sample_rate = 0.7, min_split_improvement = best_gamma1, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_de_c2c <- h2o.xgboost(
  x = X_c2b, y = Y, training_frame = track_de_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth2, learn_rate = 0.01, min_rows = best_child2, sample_rate = 0.7, min_split_improvement = best_gamma2, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_de_c3c <- h2o.xgboost(
  x = X_c3b, y = Y, training_frame = track_de_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth3, learn_rate = 0.01, min_rows = best_child3, sample_rate = 0.7, min_split_improvement = best_gamma3, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_de_c4c <- h2o.xgboost(
  x = X_c4b, y = Y, training_frame = track_de_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth4, learn_rate = 0.01, min_rows = best_child4, sample_rate = 0.7, min_split_improvement = best_gamma4, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

glm_de_c6b <- h2o.stackedEnsemble(
  y = Y, training_frame = track_de_train_h2o,
  base_models = list(xgb_de_c1c, xgb_de_c2c, xgb_de_c3c, xgb_de_c4c),
  metalearner_algorithm = "glm", metalearner_nfolds = 10)

tab <- h2o.getModel(glm_de_c6b@model$metalearner$name)@model$coefficients_table
h2o.saveModel(glm_de_c6b)

rtffile <- RTF("c_de6_stacked.doc")
addTable(rtffile, cbind(tab[,1],round(tab[,2:3], digits = 3)))
done(rtffile)

##################################################################################

# Changed - France - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_fr_c1ba <- train(",model_c1b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth1 <- xgb_fr_c1ba$results$max_depth[which.min(xgb_fr_c1ba$results[,"logLoss"])]
best_child1 <- xgb_fr_c1ba$results$min_child_weight[which.min(xgb_fr_c1ba$results[,"logLoss"])]
best_gamma1 <- xgb_fr_c1ba$results$gamma[which.min(xgb_fr_c1ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth1-1, best_depth1, best_depth1+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child1,
                        subsample = c(0.7, 1),
                        gamma = best_gamma1,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_c1b <- train(",model_c1b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_fr_c1b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_c1b <- train(",model_c1b,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_fr_c1b, metric = "ROC")

# Changed - France - categories + domains/apps

set.seed(48284)
eval(parse(text=paste("xgb_fr_c2ba <- train(",model_c2b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth2 <- xgb_fr_c2ba$results$max_depth[which.min(xgb_fr_c2ba$results[,"logLoss"])]
best_child2 <- xgb_fr_c2ba$results$min_child_weight[which.min(xgb_fr_c2ba$results[,"logLoss"])]
best_gamma2 <- xgb_fr_c2ba$results$gamma[which.min(xgb_fr_c2ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth2-1, best_depth2, best_depth2+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child2,
                        subsample = c(0.7, 1),
                        gamma = best_gamma2,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_c2b <- train(",model_c2b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_fr_c2b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_c2b <- train(",model_fr_c2b,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_fr_c2b, metric = "ROC")

# Changed - France - reduced BERT

set.seed(48284)
eval(parse(text=paste("xgb_fr_c3ba <- train(",model_c3b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth3 <- xgb_fr_c3ba$results$max_depth[which.min(xgb_fr_c3ba$results[,"logLoss"])]
best_child3 <- xgb_fr_c3ba$results$min_child_weight[which.min(xgb_fr_c3ba$results[,"logLoss"])]
best_gamma3 <- xgb_fr_c3ba$results$gamma[which.min(xgb_fr_c3ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth3-1, best_depth3, best_depth3+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child3,
                        subsample = c(0.7, 1),
                        gamma = best_gamma3,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_c3b <- train(",model_c3b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_fr_c3b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_c3b <- train(",model_c3b,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_fr_c3b, metric = "ROC")

# Changed - France - topics

set.seed(48284)
eval(parse(text=paste("xgb_fr_c4ba <- train(",model_c4b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth4 <- xgb_fr_c4ba$results$max_depth[which.min(xgb_fr_c4ba$results[,"logLoss"])]
best_child4 <- xgb_fr_c4ba$results$min_child_weight[which.min(xgb_fr_c4ba$results[,"logLoss"])]
best_gamma4 <- xgb_fr_c4ba$results$gamma[which.min(xgb_fr_c4ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth4-1, best_depth4, best_depth4+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child4,
                        subsample = c(0.7, 1),
                        gamma = best_gamma4,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_c4b <- train(",model_c4b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_fr_c4b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_c4b <- train(",model_c4b,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_fr_c4b, metric = "ROC")

# Changed - France - all

set.seed(48284)
eval(parse(text=paste("xgb_fr_c5ba <- train(",model_c5b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth5 <- xgb_fr_c5ba$results$max_depth[which.min(xgb_fr_c5ba$results[,"logLoss"])]
best_child5 <- xgb_fr_c5ba$results$min_child_weight[which.min(xgb_fr_c5ba$results[,"logLoss"])]
best_gamma5 <- xgb_fr_c5ba$results$gamma[which.min(xgb_fr_c5ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth5-1, best_depth5, best_depth5+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child5,
                        subsample = c(0.7, 1),
                        gamma = best_gamma5,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_c5b <- train(",model_c5b,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_fr_c5b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_c5b <- train(",model_c5b,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_fr_c5b, metric = "ROC")

# Changed - France - stacked model

xgb_fr_c1c <- h2o.xgboost(
  x = X_c1b, y = Y, training_frame = track_fr_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth1, learn_rate = 0.01, min_rows = best_child1, sample_rate = 0.7, min_split_improvement = best_gamma1, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_fr_c2c <- h2o.xgboost(
  x = X_c2b, y = Y, training_frame = track_fr_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth2, learn_rate = 0.01, min_rows = best_child2, sample_rate = 0.7, min_split_improvement = best_gamma2, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_fr_c3c <- h2o.xgboost(
  x = X_c3b, y = Y, training_frame = track_fr_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth3, learn_rate = 0.01, min_rows = best_child3, sample_rate = 0.7, min_split_improvement = best_gamma3, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_fr_c4c <- h2o.xgboost(
  x = X_c4b, y = Y, training_frame = track_fr_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth4, learn_rate = 0.01, min_rows = best_child4, sample_rate = 0.7, min_split_improvement = best_gamma4, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

glm_fr_c6b <- h2o.stackedEnsemble(
  y = Y, training_frame = track_fr_train_h2o,
  base_models = list(xgb_fr_c1c, xgb_fr_c2c, xgb_fr_c3c, xgb_fr_c4c),
  metalearner_algorithm = "glm", metalearner_nfolds = 10)

tab <- h2o.getModel(glm_fr_c6b@model$metalearner$name)@model$coefficients_table
h2o.saveModel(glm_fr_c6b)

rtffile <- RTF("c_fr6_stacked.doc")
addTable(rtffile, cbind(tab[,1],round(tab[,2:3], digits = 3)))
done(rtffile)

##################################################################################

# Changed - UK - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_uk_c1ba <- train(",model_c1b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth1 <- xgb_uk_c1ba$results$max_depth[which.min(xgb_uk_c1ba$results[,"logLoss"])]
best_child1 <- xgb_uk_c1ba$results$min_child_weight[which.min(xgb_uk_c1ba$results[,"logLoss"])]
best_gamma1 <- xgb_uk_c1ba$results$gamma[which.min(xgb_uk_c1ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth1-1, best_depth1, best_depth1+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child1,
                        subsample = c(0.7, 1),
                        gamma = best_gamma1,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_c1b <- train(",model_c1b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_uk_c1b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_c1b <- train(",model_c1b,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_uk_c1b, metric = "ROC")

# Changed - UK - categories + domains/apps

set.seed(48284)
eval(parse(text=paste("xgb_uk_c2ba <- train(",model_c2b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth2 <- xgb_uk_c2ba$results$max_depth[which.min(xgb_uk_c2ba$results[,"logLoss"])]
best_child2 <- xgb_uk_c2ba$results$min_child_weight[which.min(xgb_uk_c2ba$results[,"logLoss"])]
best_gamma2 <- xgb_uk_c2ba$results$gamma[which.min(xgb_uk_c2ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth2-1, best_depth2, best_depth2+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child2,
                        subsample = c(0.7, 1),
                        gamma = best_gamma2,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_c2b <- train(",model_c2b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_uk_c2b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_c2b <- train(",model_c2b,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_uk_c2b, metric = "ROC")

# Changed - UK - reduced BERT

set.seed(48284)
eval(parse(text=paste("xgb_uk_c3ba <- train(",model_c3b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth3 <- xgb_uk_c3ba$results$max_depth[which.min(xgb_uk_c3ba$results[,"logLoss"])]
best_child3 <- xgb_uk_c3ba$results$min_child_weight[which.min(xgb_uk_c3ba$results[,"logLoss"])]
best_gamma3 <- xgb_uk_c3ba$results$gamma[which.min(xgb_uk_c3ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth3-1, best_depth3, best_depth3+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child3,
                        subsample = c(0.7, 1),
                        gamma = best_gamma3,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_c3b <- train(",model_c3b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_uk_c3b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_c3b <- train(",model_c3b,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_uk_c3b, metric = "ROC")

# Changed - UK - topics

set.seed(48284)
eval(parse(text=paste("xgb_uk_c4ba <- train(",model_c4b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth4 <- xgb_uk_c4ba$results$max_depth[which.min(xgb_uk_c4ba$results[,"logLoss"])]
best_child4 <- xgb_uk_c4ba$results$min_child_weight[which.min(xgb_uk_c4ba$results[,"logLoss"])]
best_gamma4 <- xgb_uk_c4ba$results$gamma[which.min(xgb_uk_c4ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth4-1, best_depth4, best_depth4+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child4,
                        subsample = c(0.7, 1),
                        gamma = best_gamma4,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_c4b <- train(",model_c4b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_uk_c4b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_c4b <- train(",model_c4b,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_uk_c4b, metric = "ROC")

# Changed - UK - all

set.seed(48284)
eval(parse(text=paste("xgb_uk_c5ba <- train(",model_c5b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'logLoss',
                      na.action = na.omit)")))

best_depth5 <- xgb_uk_c5ba$results$max_depth[which.min(xgb_uk_c5ba$results[,"logLoss"])]
best_child5 <- xgb_uk_c5ba$results$min_child_weight[which.min(xgb_uk_c5ba$results[,"logLoss"])]
best_gamma5 <- xgb_uk_c5ba$results$gamma[which.min(xgb_uk_c5ba$results[,"logLoss"])]

xgb_grid <- expand.grid(max_depth = c(best_depth5-1, best_depth5, best_depth5+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child5,
                        subsample = c(0.7, 1),
                        gamma = best_gamma5,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_c5b <- train(",model_c5b,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(xgb_uk_c5b, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_c5b <- train(",model_c5b,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'logLoss',
                      na.action = na.omit)")))
#plot(mb_uk_c5b, metric = "ROC")

# Changed - UK - stacked model

xgb_uk_c1c <- h2o.xgboost(
  x = X_c1b, y = Y, training_frame = track_uk_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth1, learn_rate = 0.01, min_rows = best_child1, sample_rate = 0.7, min_split_improvement = best_gamma1, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_uk_c2c <- h2o.xgboost(
  x = X_c2b, y = Y, training_frame = track_uk_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth2, learn_rate = 0.01, min_rows = best_child2, sample_rate = 0.7, min_split_improvement = best_gamma2, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_uk_c3c <- h2o.xgboost(
  x = X_c3b, y = Y, training_frame = track_uk_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth3, learn_rate = 0.01, min_rows = best_child3, sample_rate = 0.7, min_split_improvement = best_gamma3, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

xgb_uk_c4c <- h2o.xgboost(
  x = X_c4b, y = Y, training_frame = track_uk_train_h2o, distribution = "bernoulli",
  ntrees = 1000, max_depth = best_depth4, learn_rate = 0.01, min_rows = best_child4, sample_rate = 0.7, min_split_improvement = best_gamma4, col_sample_rate_per_tree = 1,
  stopping_rounds = 50, stopping_metric = "AUC",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 93283)

glm_uk_c6b <- h2o.stackedEnsemble(
  y = Y, training_frame = track_uk_train_h2o,
  base_models = list(xgb_uk_c1c, xgb_uk_c2c, xgb_uk_c3c, xgb_uk_c4c),
  metalearner_algorithm = "glm", metalearner_nfolds = 10)

tab <- h2o.getModel(glm_uk_c6b@model$metalearner$name)@model$coefficients_table
h2o.saveModel(glm_uk_c6b)

rtffile <- RTF("c_uk6_stacked.doc")
addTable(rtffile, cbind(tab[,1],round(tab[,2:3], digits = 3)))
done(rtffile)

##################################################################################
# Variable Importance
##################################################################################

plot(varImp(xgb_de_c1b), top = 10)
plot(varImp(xgb_de_c2b), top = 10)
plot(varImp(xgb_de_c3b), top = 10)
plot(varImp(xgb_de_c4b), top = 10)
plot(varImp(xgb_de_c5b), top = 10)

plot(varImp(xgb_fr_c1b), top = 10)
plot(varImp(xgb_fr_c2b), top = 10)
plot(varImp(xgb_fr_c3b), top = 10)
plot(varImp(xgb_fr_c4b), top = 10)
plot(varImp(xgb_fr_c5b), top = 10)

plot(varImp(xgb_uk_c1b), top = 10)
plot(varImp(xgb_uk_c2b), top = 10)
plot(varImp(xgb_uk_c3b), top = 10)
plot(varImp(xgb_uk_c4b), top = 10)
plot(varImp(xgb_uk_c5b), top = 10)

##################################################################################
# SHAP
##################################################################################

# Changed - Germany - socio_demo + reduced BERT
xgb_de_5 <- xgb_de_c5b$finalModel
eval(parse(text=paste("x_data <- model.matrix(",model_c5b," , track_de_train)[, -1]")))
shap_long <- shap.prep(xgb_de_5, X_train = x_data, top_n = 20)
shap.plot.summary(shap_long, dilute = T)

# Changed - France - socio_demo + reduced BERT
xgb_fr_5 <- xgb_fr_c5b$finalModel
eval(parse(text=paste("x_data <- model.matrix(",model_c5b," , track_fr_train)[, -1]")))
shap_long <- shap.prep(xgb_fr_5, X_train = x_data, top_n = 20)
shap.plot.summary(shap_long, dilute = T)

# Changed - UK - socio_demo + reduced BERT
xgb_uk_5 <- xgb_uk_c5b$finalModel
eval(parse(text=paste("x_data <- model.matrix(",model_c5b," , track_uk_train)[, -1]")))
shap_long <- shap.prep(xgb_uk_5, X_train = x_data, top_n = 20)
shap.plot.summary(shap_long, dilute = T)

##################################################################################
# Compare CV performance
##################################################################################

resamps_de <- resamples(list(xgb_de_c1b, xgb_de_c2b, xgb_de_c3b,
                             xgb_de_c4b, xgb_de_c5b))

resamps_fr <- resamples(list(xgb_fr_c1b, xgb_fr_c2b, xgb_fr_c3b,
                             xgb_fr_c4b, xgb_fr_c5b))

resamps_uk <- resamples(list(xgb_uk_c1b, xgb_uk_c2b, xgb_uk_c3b,
                             xgb_uk_c4b, xgb_uk_c5b))

sum_resamps_de <- summary(resamps_de)
sum_resamps_fr <- summary(resamps_fr)
sum_resamps_uk <- summary(resamps_uk)

changed_cv_de <- data.frame("Country" = "Germany", 
                            "Model" = names(sum_resamps_de$statistics$ROC[, 4]),
                            "AUC" = sum_resamps_de$statistics$ROC[, 4])
changed_cv_de <- changed_cv_de %>% add_row(Country = "Germany", Model = "Model6", AUC = h2o.auc(glm_de_c6b, xval = T))

changed_cv_fr <- data.frame("Country" = "France", 
                            "Model" = names(sum_resamps_fr$statistics$ROC[, 4]),
                            "AUC" = sum_resamps_fr$statistics$ROC[, 4])
changed_cv_fr <- changed_cv_fr %>% add_row(Country = "France", Model = "Model6", AUC = h2o.auc(glm_fr_c6b, xval = T))

changed_cv_uk <- data.frame("Country" = "UK", 
                            "Model" = names(sum_resamps_uk$statistics$ROC[, 4]),
                            "AUC" = sum_resamps_uk$statistics$ROC[, 4])
changed_cv_uk <- changed_cv_uk %>% add_row(Country = "UK", Model = "Model6", AUC = h2o.auc(glm_uk_c6b, xval = T))

changed_cv_roc <- as.data.frame(rbind(changed_cv_de, 
                                      changed_cv_fr, 
                                      changed_cv_uk))

changed_cv_roc$Model <- fct_recode(changed_cv_roc$Model,
                                 "Model1: Demo" = "Model1",
                                 "Model2: Categories" = "Model2",
                                 "Model3: BERT" = "Model3",
                                 "Model4: Topics" = "Model4",
                                 "Model5: All" = "Model5",
                                 "Model6: Stacking" = "Model6")

##################################################################################
# Predict in test data
##################################################################################

p_xgb_de_c1b <- predict(xgb_de_c1b, newdata = track_de_test, type = "prob")
p_xgb_de_c2b <- predict(xgb_de_c2b, newdata = track_de_test, type = "prob")
p_xgb_de_c3b <- predict(xgb_de_c3b, newdata = track_de_test, type = "prob")
p_xgb_de_c4b <- predict(xgb_de_c4b, newdata = track_de_test, type = "prob")
p_xgb_de_c5b <- predict(xgb_de_c5b, newdata = track_de_test, type = "prob")
p_glm_de_c6b <- h2o.predict(glm_de_c6b, newdata = track_de_test_h2o)

p_xgb_fr_c1b <- predict(xgb_fr_c1b, newdata = track_fr_test, type = "prob")
p_xgb_fr_c2b <- predict(xgb_fr_c2b, newdata = track_fr_test, type = "prob")
p_xgb_fr_c3b <- predict(xgb_fr_c3b, newdata = track_fr_test, type = "prob")
p_xgb_fr_c4b <- predict(xgb_fr_c4b, newdata = track_fr_test, type = "prob")
p_xgb_fr_c5b <- predict(xgb_fr_c5b, newdata = track_fr_test, type = "prob")
p_glm_fr_c6b <- h2o.predict(glm_fr_c6b, newdata = track_fr_test_h2o)

p_xgb_uk_c1b <- predict(xgb_uk_c1b, newdata = track_uk_test, type = "prob")
p_xgb_uk_c2b <- predict(xgb_uk_c2b, newdata = track_uk_test, type = "prob")
p_xgb_uk_c3b <- predict(xgb_uk_c3b, newdata = track_uk_test, type = "prob")
p_xgb_uk_c4b <- predict(xgb_uk_c4b, newdata = track_uk_test, type = "prob")
p_xgb_uk_c5b <- predict(xgb_uk_c5b, newdata = track_uk_test, type = "prob")
p_glm_uk_c6b <- h2o.predict(glm_uk_c6b, newdata = track_uk_test_h2o)

roc_xgb_de_c1b <- roc(response = track_de_test$changed, predictor = p_xgb_de_c1b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_xgb_de_c2b <- roc(response = track_de_test$changed, predictor = p_xgb_de_c2b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_xgb_de_c3b <- roc(response = track_de_test$changed, predictor = p_xgb_de_c3b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_xgb_de_c4b <- roc(response = track_de_test$changed, predictor = p_xgb_de_c4b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_xgb_de_c5b <- roc(response = track_de_test$changed, predictor = p_xgb_de_c5b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_glm_de_c6b <- roc(response = track_de_test$changed, predictor = as.vector(p_glm_de_c6b$changed))$auc

changed_test_de <- data.frame("Country" = "Germany", 
                              "Model" = c("Model1: Demo", "Model2: Categories", "Model3: BERT", "Model4: Topics", "Model5: All", "Model6: Stacking"),
                              "AUC" = rbind(roc_xgb_de_c1b, roc_xgb_de_c2b, roc_xgb_de_c3b, roc_xgb_de_c4b, roc_xgb_de_c5b, roc_glm_de_c6b))

roc_xgb_fr_c1b <- roc(response = track_fr_test$changed, predictor = p_xgb_fr_c1b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_xgb_fr_c2b <- roc(response = track_fr_test$changed, predictor = p_xgb_fr_c2b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_xgb_fr_c3b <- roc(response = track_fr_test$changed, predictor = p_xgb_fr_c3b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_xgb_fr_c4b <- roc(response = track_fr_test$changed, predictor = p_xgb_fr_c4b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_xgb_fr_c5b <- roc(response = track_fr_test$changed, predictor = p_xgb_fr_c5b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_glm_fr_c6b <- roc(response = track_fr_test$changed, predictor = as.vector(p_glm_fr_c6b$changed))$auc

changed_test_fr <- data.frame("Country" = "France", 
                              "Model" = c("Model1: Demo", "Model2: Categories", "Model3: BERT", "Model4: Topics", "Model5: All", "Model6: Stacking"),
                              "AUC" = rbind(roc_xgb_fr_c1b, roc_xgb_fr_c2b, roc_xgb_fr_c3b, roc_xgb_fr_c4b, roc_xgb_fr_c5b, roc_glm_fr_c6b))

roc_xgb_uk_c1b <- roc(response = track_uk_test$changed, predictor = p_xgb_uk_c1b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_xgb_uk_c2b <- roc(response = track_uk_test$changed, predictor = p_xgb_uk_c2b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_xgb_uk_c3b <- roc(response = track_uk_test$changed, predictor = p_xgb_uk_c3b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_xgb_uk_c4b <- roc(response = track_uk_test$changed, predictor = p_xgb_uk_c4b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_xgb_uk_c5b <- roc(response = track_uk_test$changed, predictor = p_xgb_uk_c5b$changed, levels = c("changed", "not_changed"), direction = ">")$auc
roc_glm_uk_c6b <- roc(response = track_uk_test$changed, predictor = as.vector(p_glm_uk_c6b$changed))$auc

changed_test_uk <- data.frame("Country" = "UK", 
                              "Model" = c("Model1: Demo", "Model2: Categories", "Model3: BERT", "Model4: Topics", "Model5: All", "Model6: Stacking"),
                              "AUC" = rbind(roc_xgb_uk_c1b, roc_xgb_uk_c2b, roc_xgb_uk_c3b, roc_xgb_uk_c4b, roc_xgb_uk_c5b, roc_glm_uk_c6b))

changed_test_roc <- as.data.frame(rbind(changed_test_de, 
                                        changed_test_fr, 
                                        changed_test_uk))

# Combine and plot results

changed_roc <- bind_rows("CV" = changed_cv_roc, "Test" = changed_test_roc, .id = "type")

ggplot(changed_roc, aes(Model, Country)) + 
  geom_tile(aes(fill = AUC), colour = "white") + 
  geom_text(aes(label = round(AUC, 2)), size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  facet_grid(cols = vars(type)) +
  labs(x = "", y = "") +
  scale_y_discrete(limits = rev(levels(changed_roc$Country))) +
  theme(legend.position = "none",
        text = element_text(size = 15),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

p1 <- ggplot(changed_cv_roc, aes(Model, Country)) + 
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

p2 <- ggplot(changed_test_roc, aes(Model, Country)) + 
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

ggsave("c_roc.png", g, width = 9, height = 6.5)

# save.image("c_results_all2.RDATA")
