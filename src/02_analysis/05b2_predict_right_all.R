##################################################################################

library(tidyverse)
library(caret)
library(xgboost)
library(mboost)
library(pROC)
library(SHAPforxgboost)

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

model_r1 <- paste("right ~", paste(demo, collapse="+"), paste("+"), paste(media, collapse="+"))
model_r2 <- paste(model_r1, paste("+"), paste(pred_track_cat_dom_all, collapse="+"), 
                  paste("+"), paste(pred_track_cat_app_all, collapse="+"), paste("+ no_app_data"))
model_r3 <- paste(model_r1, paste("+"), paste(pred_track_dom_all, collapse="+"), 
                  paste("+"), paste(pred_track_app_all, collapse="+"), paste("+ no_app_data"))
model_r4 <- paste(model_r1, paste("+"), paste(bert_pca_cluster_all, collapse="+"), paste("+ no_bert_data"))
model_r5 <- paste(model_r1, paste("+"), paste(bert_pca_cluster_reduced_all, collapse="+"), paste("+ no_reduced_bert"))
model_r6 <- paste(model_r1, paste("+"), paste(topics_all, collapse="+"), paste("+ no_topics"))

##################################################################################
# Models - XGBoost + mboost
##################################################################################

# Right - Germany - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_de_r1a <- train(",model_r1,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_de_r1a$results$max_depth[which.max(xgb_de_r1a$results[,"ROC"])]
best_child <- xgb_de_r1a$results$min_child_weight[which.max(xgb_de_r1a$results[,"ROC"])]
best_gamma <- xgb_de_r1a$results$gamma[which.max(xgb_de_r1a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_r1 <- train(",model_r1,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_de_r1, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_r1 <- train(",model_r1,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_de_r1, metric = "ROC")

# Right - Germany - socio_demo + categories

set.seed(48284)
eval(parse(text=paste("xgb_de_r2a <- train(",model_r2,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_de_r2a$results$max_depth[which.max(xgb_de_r2a$results[,"ROC"])]
best_child <- xgb_de_r2a$results$min_child_weight[which.max(xgb_de_r2a$results[,"ROC"])]
best_gamma <- xgb_de_r2a$results$gamma[which.max(xgb_de_r2a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_r2 <- train(",model_r2,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_de_r2, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_r2 <- train(",model_r2,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_de_r2, metric = "ROC")

# Right - Germany - socio_demo + domains/apps

set.seed(48284)
eval(parse(text=paste("xgb_de_r3a <- train(",model_r3,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_de_r3a$results$max_depth[which.max(xgb_de_r3a$results[,"ROC"])]
best_child <- xgb_de_r3a$results$min_child_weight[which.max(xgb_de_r3a$results[,"ROC"])]
best_gamma <- xgb_de_r3a$results$gamma[which.max(xgb_de_r3a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_r3 <- train(",model_r3,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_de_r3, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_r3 <- train(",model_r3,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_de_r3, metric = "ROC")

# Right - Germany - socio_demo + BERT

set.seed(48284)
eval(parse(text=paste("xgb_de_r4a <- train(",model_r4,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_de_r4a$results$max_depth[which.max(xgb_de_r4a$results[,"ROC"])]
best_child <- xgb_de_r4a$results$min_child_weight[which.max(xgb_de_r4a$results[,"ROC"])]
best_gamma <- xgb_de_r4a$results$gamma[which.max(xgb_de_r4a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_r4 <- train(",model_r4,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_de_r4, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_r4 <- train(",model_r4,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_de_r4, metric = "ROC")

# Right - Germany - socio_demo + reduced BERT

set.seed(48284)
eval(parse(text=paste("xgb_de_r5a <- train(",model_r5,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_de_r5a$results$max_depth[which.max(xgb_de_r5a$results[,"ROC"])]
best_child <- xgb_de_r5a$results$min_child_weight[which.max(xgb_de_r5a$results[,"ROC"])]
best_gamma <- xgb_de_r5a$results$gamma[which.max(xgb_de_r5a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_r5 <- train(",model_r5,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_de_r5, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_r5 <- train(",model_r5,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_de_r5, metric = "ROC")

# Right - Germany - socio_demo + topics

set.seed(48284)
eval(parse(text=paste("xgb_de_r6a <- train(",model_r6,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_de_r6a$results$max_depth[which.max(xgb_de_r6a$results[,"ROC"])]
best_child <- xgb_de_r6a$results$min_child_weight[which.max(xgb_de_r6a$results[,"ROC"])]
best_gamma <- xgb_de_r6a$results$gamma[which.max(xgb_de_r6a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_r6 <- train(",model_r6,",
                      data = track_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_de_r6, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_r6 <- train(",model_r6,",
                      data = track_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_de_r6, metric = "ROC")

##################################################################################

# Right - France - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_fr_r1a <- train(",model_r1,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_r1a$results$max_depth[which.max(xgb_fr_r1a$results[,"ROC"])]
best_child <- xgb_fr_r1a$results$min_child_weight[which.max(xgb_fr_r1a$results[,"ROC"])]
best_gamma <- xgb_fr_r1a$results$gamma[which.max(xgb_fr_r1a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_r1 <- train(",model_r1,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_fr_r1, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_r1 <- train(",model_r1,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_fr_r1, metric = "ROC")

# Right - France - socio_demo + categories

set.seed(48284)
eval(parse(text=paste("xgb_fr_r2a <- train(",model_r2,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_r2a$results$max_depth[which.max(xgb_fr_r2a$results[,"ROC"])]
best_child <- xgb_fr_r2a$results$min_child_weight[which.max(xgb_fr_r2a$results[,"ROC"])]
best_gamma <- xgb_fr_r2a$results$gamma[which.max(xgb_fr_r2a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_r2 <- train(",model_r2,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_fr_r2, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_r2 <- train(",model_r2,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_fr_r2, metric = "ROC")

# Right - France - socio_demo + domains/apps

set.seed(48284)
eval(parse(text=paste("xgb_fr_r3a <- train(",model_r3,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_r3a$results$max_depth[which.max(xgb_fr_r3a$results[,"ROC"])]
best_child <- xgb_fr_r3a$results$min_child_weight[which.max(xgb_fr_r3a$results[,"ROC"])]
best_gamma <- xgb_fr_r3a$results$gamma[which.max(xgb_fr_r3a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_r3 <- train(",model_r3,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_fr_r3, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_r3 <- train(",model_r3,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_fr_r3, metric = "ROC")

# Right - France - socio_demo + BERT

set.seed(48284)
eval(parse(text=paste("xgb_fr_r4a <- train(",model_r4,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_r4a$results$max_depth[which.max(xgb_fr_r4a$results[,"ROC"])]
best_child <- xgb_fr_r4a$results$min_child_weight[which.max(xgb_fr_r4a$results[,"ROC"])]
best_gamma <- xgb_fr_r4a$results$gamma[which.max(xgb_fr_r4a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_r4 <- train(",model_r4,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_fr_r4, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_r4 <- train(",model_r4,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_fr_r4, metric = "ROC")

# Right - France - socio_demo + reduced BERT

set.seed(48284)
eval(parse(text=paste("xgb_fr_r5a <- train(",model_r5,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_r5a$results$max_depth[which.max(xgb_fr_r5a$results[,"ROC"])]
best_child <- xgb_fr_r5a$results$min_child_weight[which.max(xgb_fr_r5a$results[,"ROC"])]
best_gamma <- xgb_fr_r5a$results$gamma[which.max(xgb_fr_r5a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_r5 <- train(",model_r5,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_fr_r5, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_r5 <- train(",model_r5,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_fr_r5, metric = "ROC")

# Right - France - socio_demo + topics

set.seed(48284)
eval(parse(text=paste("xgb_fr_r6a <- train(",model_r6,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_r6a$results$max_depth[which.max(xgb_fr_r6a$results[,"ROC"])]
best_child <- xgb_fr_r6a$results$min_child_weight[which.max(xgb_fr_r6a$results[,"ROC"])]
best_gamma <- xgb_fr_r6a$results$gamma[which.max(xgb_fr_r6a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_r6 <- train(",model_r6,",
                      data = track_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_fr_r6, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_r6 <- train(",model_r6,",
                      data = track_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_fr_r6, metric = "ROC")

##################################################################################

# Right - UK - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_uk_r1a <- train(",model_r1,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_r1a$results$max_depth[which.max(xgb_uk_r1a$results[,"ROC"])]
best_child <- xgb_uk_r1a$results$min_child_weight[which.max(xgb_uk_r1a$results[,"ROC"])]
best_gamma <- xgb_uk_r1a$results$gamma[which.max(xgb_uk_r1a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_r1 <- train(",model_r1,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_uk_r1, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_r1 <- train(",model_r1,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_uk_r1, metric = "ROC")

# Right - UK - socio_demo + categories

set.seed(48284)
eval(parse(text=paste("xgb_uk_r2a <- train(",model_r2,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_r2a$results$max_depth[which.max(xgb_uk_r2a$results[,"ROC"])]
best_child <- xgb_uk_r2a$results$min_child_weight[which.max(xgb_uk_r2a$results[,"ROC"])]
best_gamma <- xgb_uk_r2a$results$gamma[which.max(xgb_uk_r2a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_r2 <- train(",model_r2,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_uk_r2, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_r2 <- train(",model_r2,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_uk_r2, metric = "ROC")

# Right - UK - socio_demo + domains/apps

set.seed(48284)
eval(parse(text=paste("xgb_uk_r3a <- train(",model_r3,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_r3a$results$max_depth[which.max(xgb_uk_r3a$results[,"ROC"])]
best_child <- xgb_uk_r3a$results$min_child_weight[which.max(xgb_uk_r3a$results[,"ROC"])]
best_gamma <- xgb_uk_r3a$results$gamma[which.max(xgb_uk_r3a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_r3 <- train(",model_r3,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_uk_r3, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_r3 <- train(",model_r3,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_uk_r3, metric = "ROC")

# Right - UK - socio_demo + BERT

set.seed(48284)
eval(parse(text=paste("xgb_uk_r4a <- train(",model_r4,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_r4a$results$max_depth[which.max(xgb_uk_r4a$results[,"ROC"])]
best_child <- xgb_uk_r4a$results$min_child_weight[which.max(xgb_uk_r4a$results[,"ROC"])]
best_gamma <- xgb_uk_r4a$results$gamma[which.max(xgb_uk_r4a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_r4 <- train(",model_r4,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_uk_r4, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_r4 <- train(",model_r4,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_uk_r4, metric = "ROC")

# Right - UK - socio_demo + reduced BERT

set.seed(48284)
eval(parse(text=paste("xgb_uk_r5a <- train(",model_r5,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_r5a$results$max_depth[which.max(xgb_uk_r5a$results[,"ROC"])]
best_child <- xgb_uk_r5a$results$min_child_weight[which.max(xgb_uk_r5a$results[,"ROC"])]
best_gamma <- xgb_uk_r5a$results$gamma[which.max(xgb_uk_r5a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_r5 <- train(",model_r5,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_uk_r5, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_r5 <- train(",model_r5,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_uk_r5, metric = "ROC")

# Right - UK - socio_demo + topics

set.seed(48284)
eval(parse(text=paste("xgb_uk_r6a <- train(",model_r6,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_r6a$results$max_depth[which.max(xgb_uk_r6a$results[,"ROC"])]
best_child <- xgb_uk_r6a$results$min_child_weight[which.max(xgb_uk_r6a$results[,"ROC"])]
best_gamma <- xgb_uk_r6a$results$gamma[which.max(xgb_uk_r6a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_r6 <- train(",model_r6,",
                      data = track_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(xgb_uk_r6, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_r6 <- train(",model_r6,",
                      data = track_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
#plot(mb_uk_r6, metric = "ROC")

##################################################################################
# Variable Importance
##################################################################################

plot(varImp(xgb_de_r1), top = 10)
plot(varImp(xgb_de_r2), top = 10)
plot(varImp(xgb_de_r3), top = 10)
plot(varImp(xgb_de_r4), top = 10)
plot(varImp(xgb_de_r5), top = 10)
plot(varImp(xgb_de_r6), top = 10)

plot(varImp(xgb_fr_r1), top = 10)
plot(varImp(xgb_fr_r2), top = 10)
plot(varImp(xgb_fr_r3), top = 10)
plot(varImp(xgb_fr_r4), top = 10)
plot(varImp(xgb_fr_r5), top = 10)
plot(varImp(xgb_fr_r6), top = 10)

plot(varImp(xgb_uk_r1), top = 10)
plot(varImp(xgb_uk_r2), top = 10)
plot(varImp(xgb_uk_r3), top = 10)
plot(varImp(xgb_uk_r4), top = 10)
plot(varImp(xgb_uk_r5), top = 10)
plot(varImp(xgb_uk_r6), top = 10)

##################################################################################
# SHAP
##################################################################################

# Right - Germany - socio_demo + reduced BERT
xgb_de_5 <- xgb_de_r5$finalModel
eval(parse(text=paste("x_data <- model.matrix(",model_r5," , track_de_train)[, -1]")))
shap_long <- shap.prep(xgb_de_5, X_train = x_data, top_n = 20)
shap.plot.summary(shap_long, dilute = T)

# Right - France - socio_demo + reduced BERT
xgb_fr_5 <- xgb_fr_r5$finalModel
eval(parse(text=paste("x_data <- model.matrix(",model_r5," , track_fr_train)[, -1]")))
shap_long <- shap.prep(xgb_fr_5, X_train = x_data, top_n = 20)
shap.plot.summary(shap_long, dilute = T)

# Right - UK - socio_demo + reduced BERT
xgb_uk_5 <- xgb_uk_r5$finalModel
eval(parse(text=paste("x_data <- model.matrix(",model_r5," , track_uk_train)[, -1]")))
shap_long <- shap.prep(xgb_uk_5, X_train = x_data, top_n = 20)
shap.plot.summary(shap_long, dilute = T)

##################################################################################
# Compare CV performance
##################################################################################

resamps_de <- resamples(list(xgb_de_r1, xgb_de_r2, xgb_de_r3,
                             xgb_de_r4, xgb_de_r5, xgb_de_r6))

resamps_fr <- resamples(list(xgb_fr_r1, xgb_fr_r2, xgb_fr_r3,
                             xgb_fr_r4, xgb_fr_r5, xgb_fr_r6))

resamps_uk <- resamples(list(xgb_uk_r1, xgb_uk_r2, xgb_uk_r3,
                             xgb_uk_r4, xgb_uk_r5, xgb_uk_r6))

sum_resamps_de <- summary(resamps_de)
sum_resamps_fr <- summary(resamps_fr)
sum_resamps_uk <- summary(resamps_uk)

right_cv_de <- data.frame("Country" = "Germany",
                         "Model" = names(sum_resamps_de$statistics$ROC[, 4]),
                         "AUC" = sum_resamps_de$statistics$ROC[, 4])

right_cv_fr <- data.frame("Country" = "France",
                         "Model" = names(sum_resamps_fr$statistics$ROC[, 4]),
                         "AUC" = sum_resamps_fr$statistics$ROC[, 4])

right_cv_uk <- data.frame("Country" = "UK",
                         "Model" = names(sum_resamps_uk$statistics$ROC[, 4]),
                         "AUC" = sum_resamps_uk$statistics$ROC[, 4])

right_cv_roc <- as.data.frame(rbind(right_cv_de, 
                                    right_cv_fr, 
                                    right_cv_uk))

##################################################################################
# Predict in test data
##################################################################################

p_xgb_de_r1 <- predict(xgb_de_r1, newdata = track_de_test, type = "prob")
p_xgb_de_r2 <- predict(xgb_de_r2, newdata = track_de_test, type = "prob")
p_xgb_de_r3 <- predict(xgb_de_r3, newdata = track_de_test, type = "prob")
p_xgb_de_r4 <- predict(xgb_de_r4, newdata = track_de_test, type = "prob")
p_xgb_de_r5 <- predict(xgb_de_r5, newdata = track_de_test, type = "prob")
p_xgb_de_r6 <- predict(xgb_de_r6, newdata = track_de_test, type = "prob")

p_xgb_fr_r1 <- predict(xgb_fr_r1, newdata = track_fr_test, type = "prob")
p_xgb_fr_r2 <- predict(xgb_fr_r2, newdata = track_fr_test, type = "prob")
p_xgb_fr_r3 <- predict(xgb_fr_r3, newdata = track_fr_test, type = "prob")
p_xgb_fr_r4 <- predict(xgb_fr_r4, newdata = track_fr_test, type = "prob")
p_xgb_fr_r5 <- predict(xgb_fr_r5, newdata = track_fr_test, type = "prob")
p_xgb_fr_r6 <- predict(xgb_fr_r6, newdata = track_fr_test, type = "prob")

p_xgb_uk_r1 <- predict(xgb_uk_r1, newdata = track_uk_test, type = "prob")
p_xgb_uk_r2 <- predict(xgb_uk_r2, newdata = track_uk_test, type = "prob")
p_xgb_uk_r3 <- predict(xgb_uk_r3, newdata = track_uk_test, type = "prob")
p_xgb_uk_r4 <- predict(xgb_uk_r4, newdata = track_uk_test, type = "prob")
p_xgb_uk_r5 <- predict(xgb_uk_r5, newdata = track_uk_test, type = "prob")
p_xgb_uk_r6 <- predict(xgb_uk_r6, newdata = track_uk_test, type = "prob")

roc_xgb_de_r1 <- roc(response = track_de_test$right, predictor = p_xgb_de_r1$right)$auc
roc_xgb_de_r2 <- roc(response = track_de_test$right, predictor = p_xgb_de_r2$right)$auc
roc_xgb_de_r3 <- roc(response = track_de_test$right, predictor = p_xgb_de_r3$right)$auc
roc_xgb_de_r4 <- roc(response = track_de_test$right, predictor = p_xgb_de_r4$right)$auc
roc_xgb_de_r5 <- roc(response = track_de_test$right, predictor = p_xgb_de_r5$right)$auc
roc_xgb_de_r6 <- roc(response = track_de_test$right, predictor = p_xgb_de_r6$right)$auc

right_test_de <- data.frame("Country" = "Germany",
                            "Model" = c("Model1", "Model2", "Model3", "Model4", "Model5", "Model6"),
                            "AUC" = rbind(roc_xgb_de_r1, roc_xgb_de_r2, roc_xgb_de_r3, roc_xgb_de_r4, roc_xgb_de_r5, roc_xgb_de_r6))

roc_xgb_fr_r1 <- roc(response = track_fr_test$right, predictor = p_xgb_fr_r1$right)$auc
roc_xgb_fr_r2 <- roc(response = track_fr_test$right, predictor = p_xgb_fr_r2$right)$auc
roc_xgb_fr_r3 <- roc(response = track_fr_test$right, predictor = p_xgb_fr_r3$right)$auc
roc_xgb_fr_r4 <- roc(response = track_fr_test$right, predictor = p_xgb_fr_r4$right)$auc
roc_xgb_fr_r5 <- roc(response = track_fr_test$right, predictor = p_xgb_fr_r5$right)$auc
roc_xgb_fr_r6 <- roc(response = track_fr_test$right, predictor = p_xgb_fr_r6$right)$auc

right_test_fr <- data.frame("Country" = "France",
                            "Model" = c("Model1", "Model2", "Model3", "Model4", "Model5", "Model6"),
                            "AUC" = rbind(roc_xgb_fr_r1, roc_xgb_fr_r2, roc_xgb_fr_r3, roc_xgb_fr_r4, roc_xgb_fr_r5, roc_xgb_fr_r6))

roc_xgb_uk_r1 <- roc(response = track_uk_test$right, predictor = p_xgb_uk_r1$right)$auc
roc_xgb_uk_r2 <- roc(response = track_uk_test$right, predictor = p_xgb_uk_r2$right)$auc
roc_xgb_uk_r3 <- roc(response = track_uk_test$right, predictor = p_xgb_uk_r3$right)$auc
roc_xgb_uk_r4 <- roc(response = track_uk_test$right, predictor = p_xgb_uk_r4$right)$auc
roc_xgb_uk_r5 <- roc(response = track_uk_test$right, predictor = p_xgb_uk_r5$right)$auc
roc_xgb_uk_r6 <- roc(response = track_uk_test$right, predictor = p_xgb_uk_r6$right)$auc

right_test_uk <- data.frame("Country" = "UK",
                            "Model" = c("Model1", "Model2", "Model3", "Model4", "Model5", "Model6"),
                            "AUC" = rbind(roc_xgb_uk_r1, roc_xgb_uk_r2, roc_xgb_uk_r3, roc_xgb_uk_r4, roc_xgb_uk_r5, roc_xgb_uk_r6))

right_test_roc <- as.data.frame(rbind(right_test_de, 
                                      right_test_fr, 
                                      right_test_uk))

# Combine and plot results

right_roc <- bind_rows("CV" = right_cv_roc, "Test" = right_test_roc, .id = "type")

ggplot(right_roc, aes(Model, Country)) + 
  geom_tile(aes(fill = AUC), colour = "white") + 
  geom_text(aes(label = round(AUC, 2)), size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  facet_grid(cols = vars(type)) +
  labs(x = "", y = "") +
  scale_y_discrete(limits = rev(levels(right_roc$Country))) +
  theme(legend.position = "none",
        text = element_text(size = 15),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

ggsave("r_roc.png", width = 8, height = 6)

# save.image("r_results_all.RDATA")
