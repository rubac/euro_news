##################################################################################

library(tidyverse)
library(caret)
library(xgboost)
library(mboost)
library(pROC)

# Set path
setwd("/home/r_uma_2019/respondi_eu/")

##################################################################################
# Setup
##################################################################################

# load data
load("./data/work/prep_predict.Rdata")

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

##################################################################################
# Models - XGBoost + mboost
##################################################################################

# Changed - Germany

model_de_c1 <- paste("d_change ~", paste(demo, collapse="+"))
model_de_c2 <- paste(model_de_c1, paste("+"), paste(pred_track_cat_dom_de, collapse="+"))
model_de_c3 <- paste(model_de_c2, paste("+"), paste(pred_track_dom_de, collapse="+"))
model_de_c4 <- paste(model_de_c1, paste("+"), paste(pred_track_cat_app_de, collapse="+"))
model_de_c5 <- paste(model_de_c4, paste("+"), paste(pred_track_app_de, collapse="+"))
model_de_c6 <- paste(model_de_c1, paste("+"), paste(pred_track_bert_de, collapse="+"))

# Changed - Germany - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_de_c1a <- train(",model_de_c1,",
                      data = track_dom_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_de_c1a$results$max_depth[which.max(xgb_de_c1a$results[,"ROC"])]
best_child <- xgb_de_c1a$results$min_child_weight[which.max(xgb_de_c1a$results[,"ROC"])]
best_gamma <- xgb_de_c1a$results$gamma[which.max(xgb_de_c1a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_c1 <- train(",model_de_c1,",
                      data = track_dom_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_de_c1, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_c1 <- train(",model_de_c1,",
                      data = track_dom_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_de_c1, metric = "ROC")

# Changed - Germany - socio_demo + domain-categories

set.seed(48284)
eval(parse(text=paste("xgb_de_c2a <- train(",model_de_c2,",
                      data = track_dom_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_de_c2a$results$max_depth[which.max(xgb_de_c2a$results[,"ROC"])]
best_child <- xgb_de_c2a$results$min_child_weight[which.max(xgb_de_c2a$results[,"ROC"])]
best_gamma <- xgb_de_c2a$results$gamma[which.max(xgb_de_c2a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_c2 <- train(",model_de_c2,",
                      data = track_dom_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_de_c2, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_c2 <- train(",model_de_c2,",
                      data = track_dom_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_de_c2, metric = "ROC")

# Changed - Germany - socio_demo + domain-categories + domains

set.seed(48284)
eval(parse(text=paste("xgb_de_c3a <- train(",model_de_c3,",
                      data = track_dom_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_de_c3a$results$max_depth[which.max(xgb_de_c3a$results[,"ROC"])]
best_child <- xgb_de_c3a$results$min_child_weight[which.max(xgb_de_c3a$results[,"ROC"])]
best_gamma <- xgb_de_c3a$results$gamma[which.max(xgb_de_c3a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_c3 <- train(",model_de_c3,",
                      data = track_dom_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_de_c3, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_c3 <- train(",model_de_c3,",
                      data = track_dom_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_de_c3, metric = "ROC")

# Changed - Germany - socio_demo + app-categories

set.seed(48284)
eval(parse(text=paste("xgb_de_c4a <- train(",model_de_c4,",
                      data = track_app_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_de_c4a$results$max_depth[which.max(xgb_de_c4a$results[,"ROC"])]
best_child <- xgb_de_c4a$results$min_child_weight[which.max(xgb_de_c4a$results[,"ROC"])]
best_gamma <- xgb_de_c4a$results$gamma[which.max(xgb_de_c4a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_c4 <- train(",model_de_c4,",
                      data = track_app_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_de_c4, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_c4 <- train(",model_de_c4,",
                      data = track_app_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_de_c4, metric = "ROC")

# Changed - Germany - socio_demo + app-categories + apps

set.seed(48284)
eval(parse(text=paste("xgb_de_c5a <- train(",model_de_c5,",
                      data = track_app_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_de_c5a$results$max_depth[which.max(xgb_de_c5a$results[,"ROC"])]
best_child <- xgb_de_c5a$results$min_child_weight[which.max(xgb_de_c5a$results[,"ROC"])]
best_gamma <- xgb_de_c5a$results$gamma[which.max(xgb_de_c5a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_c5 <- train(",model_de_c5,",
                      data = track_app_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_de_c5, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_c5 <- train(",model_de_c5,",
                      data = track_app_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_de_c5, metric = "ROC")

# Changed - Germany - socio_demo + bert

set.seed(48284)
eval(parse(text=paste("xgb_de_c6a <- train(",model_de_c6,",
                      data = track_bert_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_de_c6a$results$max_depth[which.max(xgb_de_c6a$results[,"ROC"])]
best_child <- xgb_de_c6a$results$min_child_weight[which.max(xgb_de_c6a$results[,"ROC"])]
best_gamma <- xgb_de_c6a$results$gamma[which.max(xgb_de_c6a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_c6 <- train(",model_de_c6,",
                      data = track_bert_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_de_c6, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_de_c6 <- train(",model_de_c6,",
                      data = track_bert_de_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_de_c6, metric = "ROC")

##################################################################################

# Changed - France

model_fr_c1 <- paste("d_change ~", paste(demo, collapse="+"))
model_fr_c2 <- paste(model_fr_c1, paste("+"), paste(pred_track_cat_dom_fr, collapse="+"))
model_fr_c3 <- paste(model_fr_c2, paste("+"), paste(pred_track_dom_fr, collapse="+"))
model_fr_c4 <- paste(model_fr_c1, paste("+"), paste(pred_track_cat_app_fr, collapse="+"))
model_fr_c5 <- paste(model_fr_c4, paste("+"), paste(pred_track_app_fr, collapse="+"))
model_fr_c6 <- paste(model_fr_c1, paste("+"), paste(pred_track_bert_fr, collapse="+"))

# Changed - France - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_fr_c1a <- train(",model_fr_c1,",
                      data = track_dom_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_c1a$results$max_depth[which.max(xgb_fr_c1a$results[,"ROC"])]
best_child <- xgb_fr_c1a$results$min_child_weight[which.max(xgb_fr_c1a$results[,"ROC"])]
best_gamma <- xgb_fr_c1a$results$gamma[which.max(xgb_fr_c1a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_c1 <- train(",model_fr_c1,",
                      data = track_dom_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_fr_c1, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_c1 <- train(",model_fr_c1,",
                      data = track_dom_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_fr_c1, metric = "ROC")

# Changed - France - socio_demo + domain-categories

set.seed(48284)
eval(parse(text=paste("xgb_fr_c2a <- train(",model_fr_c2,",
                      data = track_dom_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_c2a$results$max_depth[which.max(xgb_fr_c2a$results[,"ROC"])]
best_child <- xgb_fr_c2a$results$min_child_weight[which.max(xgb_fr_c2a$results[,"ROC"])]
best_gamma <- xgb_fr_c2a$results$gamma[which.max(xgb_fr_c2a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_c2 <- train(",model_fr_c2,",
                      data = track_dom_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_fr_c2, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_c2 <- train(",model_fr_c2,",
                      data = track_dom_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_fr_c2, metric = "ROC")

# Changed - France - socio_demo + domain-categories + domains

set.seed(48284)
eval(parse(text=paste("xgb_fr_c3a <- train(",model_fr_c3,",
                      data = track_dom_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_c3a$results$max_depth[which.max(xgb_fr_c3a$results[,"ROC"])]
best_child <- xgb_fr_c3a$results$min_child_weight[which.max(xgb_fr_c3a$results[,"ROC"])]
best_gamma <- xgb_fr_c3a$results$gamma[which.max(xgb_fr_c3a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_c3 <- train(",model_fr_c3,",
                      data = track_dom_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_fr_c3, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_c3 <- train(",model_fr_c3,",
                      data = track_dom_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_fr_c3, metric = "ROC")

# Changed - France - socio_demo + app-categories

set.seed(48284)
eval(parse(text=paste("xgb_fr_c4a <- train(",model_fr_c4,",
                      data = track_app_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_c4a$results$max_depth[which.max(xgb_fr_c4a$results[,"ROC"])]
best_child <- xgb_fr_c4a$results$min_child_weight[which.max(xgb_fr_c4a$results[,"ROC"])]
best_gamma <- xgb_fr_c4a$results$gamma[which.max(xgb_fr_c4a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_c4 <- train(",model_fr_c4,",
                      data = track_app_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_fr_c4, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_c4 <- train(",model_fr_c4,",
                      data = track_app_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_fr_c4, metric = "ROC")

# Changed - France - socio_demo + app-categories + apps

set.seed(48284)
eval(parse(text=paste("xgb_fr_c5a <- train(",model_fr_c5,",
                      data = track_app_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_c5a$results$max_depth[which.max(xgb_fr_c5a$results[,"ROC"])]
best_child <- xgb_fr_c5a$results$min_child_weight[which.max(xgb_fr_c5a$results[,"ROC"])]
best_gamma <- xgb_fr_c5a$results$gamma[which.max(xgb_fr_c5a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_c5 <- train(",model_fr_c5,",
                      data = track_app_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_fr_c5, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_c5 <- train(",model_fr_c5,",
                      data = track_app_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_fr_c5, metric = "ROC")

# Changed - France - socio_demo + bert

set.seed(48284)
eval(parse(text=paste("xgb_fr_c6a <- train(",model_fr_c6,",
                      data = track_bert_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_c6a$results$max_depth[which.max(xgb_fr_c6a$results[,"ROC"])]
best_child <- xgb_fr_c6a$results$min_child_weight[which.max(xgb_fr_c6a$results[,"ROC"])]
best_gamma <- xgb_fr_c6a$results$gamma[which.max(xgb_fr_c6a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_c6 <- train(",model_fr_c6,",
                      data = track_bert_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_fr_c6, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_fr_c6 <- train(",model_fr_c6,",
                      data = track_bert_fr_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_fr_c6, metric = "ROC")

##################################################################################

# Changed - UK

model_uk_c1 <- paste("d_change ~", paste(demo, collapse="+"))
model_uk_c2 <- paste(model_uk_c1, paste("+"), paste(pred_track_cat_dom_uk, collapse="+"))
model_uk_c3 <- paste(model_uk_c2, paste("+"), paste(pred_track_dom_uk, collapse="+"))
model_uk_c4 <- paste(model_uk_c1, paste("+"), paste(pred_track_cat_app_uk, collapse="+"))
model_uk_c5 <- paste(model_uk_c4, paste("+"), paste(pred_track_app_uk, collapse="+"))
model_uk_c6 <- paste(model_uk_c1, paste("+"), paste(pred_track_bert_uk, collapse="+"))

# Changed - UK - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_uk_c1a <- train(",model_uk_c1,",
                      data = track_dom_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_c1a$results$max_depth[which.max(xgb_uk_c1a$results[,"ROC"])]
best_child <- xgb_uk_c1a$results$min_child_weight[which.max(xgb_uk_c1a$results[,"ROC"])]
best_gamma <- xgb_uk_c1a$results$gamma[which.max(xgb_uk_c1a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_c1 <- train(",model_uk_c1,",
                      data = track_dom_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_uk_c1, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_c1 <- train(",model_uk_c1,",
                      data = track_dom_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_uk_c1, metric = "ROC")

# Changed - UK - socio_demo + domain-categories

set.seed(48284)
eval(parse(text=paste("xgb_uk_c2a <- train(",model_uk_c2,",
                      data = track_dom_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_c2a$results$max_depth[which.max(xgb_uk_c2a$results[,"ROC"])]
best_child <- xgb_uk_c2a$results$min_child_weight[which.max(xgb_uk_c2a$results[,"ROC"])]
best_gamma <- xgb_uk_c2a$results$gamma[which.max(xgb_uk_c2a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_c2 <- train(",model_uk_c2,",
                      data = track_dom_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_uk_c2, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_c2 <- train(",model_uk_c2,",
                      data = track_dom_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_uk_c2, metric = "ROC")

# Changed - UK - socio_demo + domain-categories + domains

set.seed(48284)
eval(parse(text=paste("xgb_uk_c3a <- train(",model_uk_c3,",
                      data = track_dom_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_c3a$results$max_depth[which.max(xgb_uk_c3a$results[,"ROC"])]
best_child <- xgb_uk_c3a$results$min_child_weight[which.max(xgb_uk_c3a$results[,"ROC"])]
best_gamma <- xgb_uk_c3a$results$gamma[which.max(xgb_uk_c3a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_c3 <- train(",model_uk_c3,",
                      data = track_dom_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_uk_c3, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_c3 <- train(",model_uk_c3,",
                      data = track_dom_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_uk_c3, metric = "ROC")

# Changed - UK - socio_demo + app-categories

set.seed(48284)
eval(parse(text=paste("xgb_uk_c4a <- train(",model_uk_c4,",
                      data = track_app_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_c4a$results$max_depth[which.max(xgb_uk_c4a$results[,"ROC"])]
best_child <- xgb_uk_c4a$results$min_child_weight[which.max(xgb_uk_c4a$results[,"ROC"])]
best_gamma <- xgb_uk_c4a$results$gamma[which.max(xgb_uk_c4a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_c4 <- train(",model_uk_c4,",
                      data = track_app_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_uk_c4, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_c4 <- train(",model_uk_c4,",
                      data = track_app_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_uk_c4, metric = "ROC")

# Changed - UK - socio_demo + app-categories + apps

set.seed(48284)
eval(parse(text=paste("xgb_uk_c5a <- train(",model_uk_c5,",
                      data = track_app_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_c5a$results$max_depth[which.max(xgb_uk_c5a$results[,"ROC"])]
best_child <- xgb_uk_c5a$results$min_child_weight[which.max(xgb_uk_c5a$results[,"ROC"])]
best_gamma <- xgb_uk_c5a$results$gamma[which.max(xgb_uk_c5a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_c5 <- train(",model_uk_c5,",
                      data = track_app_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_uk_c5, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_c5 <- train(",model_uk_c5,",
                      data = track_app_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_uk_c5, metric = "ROC")

# Changed - UK - socio_demo + bert

set.seed(48284)
eval(parse(text=paste("xgb_uk_c6a <- train(",model_uk_c6,",
                      data = track_bert_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'ROC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_c6a$results$max_depth[which.max(xgb_uk_c6a$results[,"ROC"])]
best_child <- xgb_uk_c6a$results$min_child_weight[which.max(xgb_uk_c6a$results[,"ROC"])]
best_gamma <- xgb_uk_c6a$results$gamma[which.max(xgb_uk_c6a$results[,"ROC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_c6 <- train(",model_uk_c6,",
                      data = track_bert_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(xgb_uk_c6, metric = "ROC")

set.seed(29483)
eval(parse(text=paste("mb_uk_c6 <- train(",model_uk_c6,",
                      data = track_bert_uk_train,
                      method = 'glmboost',
                      trControl = ctrl,
                      tuneGrid = mboost_grid,
                      metric = 'ROC',
                      na.action = na.omit)")))
plot(mb_uk_c6, metric = "ROC")

##################################################################################
# Variable Importance
##################################################################################

plot(varImp(xgb_de_c1), top = 10)
plot(varImp(xgb_de_c2), top = 10)
plot(varImp(xgb_de_c3), top = 10)
plot(varImp(xgb_de_c4), top = 10)
plot(varImp(xgb_de_c5), top = 10)
plot(varImp(xgb_de_c6), top = 10)

varImp(xgb_de_c6)$importance %>% 
  as.data.frame() %>%
  rownames_to_column() %>%
  arrange(Overall) %>%
  top_n(15, Overall) %>% 
  mutate(rowname = forcats::fct_inorder(rowname )) %>%
  ggplot() +
  geom_col(aes(x = rowname, y = Overall)) +
  labs(y = "Importance", x = "") +
  coord_flip()
# ggsave("imp_de_c.png", width = 6, height = 6)

plot(varImp(xgb_fr_c1), top = 10)
plot(varImp(xgb_fr_c2), top = 10)
plot(varImp(xgb_fr_c3), top = 10)
plot(varImp(xgb_fr_c4), top = 10)
plot(varImp(xgb_fr_c5), top = 10)
plot(varImp(xgb_fr_c6), top = 10)

varImp(xgb_fr_c6)$importance %>% 
  as.data.frame() %>%
  rownames_to_column() %>%
  arrange(Overall) %>%
  top_n(15, Overall) %>% 
  mutate(rowname = forcats::fct_inorder(rowname )) %>%
  ggplot() +
  geom_col(aes(x = rowname, y = Overall)) +
  labs(y = "Importance", x = "") +
  coord_flip()
# ggsave("imp_fr_c.png", width = 6, height = 6)

plot(varImp(xgb_uk_c1), top = 10)
plot(varImp(xgb_uk_c2), top = 10)
plot(varImp(xgb_uk_c3), top = 10)
plot(varImp(xgb_uk_c4), top = 10)
plot(varImp(xgb_uk_c5), top = 10)
plot(varImp(xgb_uk_c6), top = 10)

varImp(xgb_uk_c6)$importance %>% 
  as.data.frame() %>%
  rownames_to_column() %>%
  arrange(Overall) %>%
  top_n(15, Overall) %>% 
  mutate(rowname = forcats::fct_inorder(rowname )) %>%
  ggplot() +
  geom_col(aes(x = rowname, y = Overall)) +
  labs(y = "Importance", x = "") +
  coord_flip()
# ggsave("imp_uk_c.png", width = 6, height = 6)

##################################################################################
# PDPs
##################################################################################

##################################################################################
# Compare CV performance
##################################################################################

resamps_de <- resamples(list(xgb_de_c1, xgb_de_c2, xgb_de_c3,
                             xgb_de_c4, xgb_de_c5, xgb_de_c6))

resamp_de <- reshape(resamps_de$values,
                     direction = "long",
                     varying = 2:ncol(resamps_de$values),
                     sep = "~",
                     v.names = c("Accuracy", "Kappa", "logLoss", "ROC", "Sens", "Spec"),
                     timevar = "model")

resamp_de <- 
  resamp_de %>%
  mutate(model = factor(model)) %>%
  mutate(model = fct_recode(model,
                            "Demo" = "1",
                            "Demo+domain_categories" = "2",
                            "Demo+domains" = "3",
                            "Demo+app_categories" = "4",
                            "Demo+apps" = "5",
                            "Demo+bert" = "6"))

#grey.colors(3)
ggplot(resamp_de) +
  geom_boxplot(aes(y = ROC, x = fct_rev(model), fill = model)) +
  ylim(0, 1) +
  labs(x = "") +
  labs(y = "ROC-AUC") +
  coord_flip() + 
  theme_bw() +
  theme(legend.position = "none") +
  theme(text = element_text(size = 20))
# ggsave("resamp_de_c.png", width = 7.5, height = 7)

resamps_fr <- resamples(list(xgb_fr_c1, xgb_fr_c2, xgb_fr_c3,
                             xgb_fr_c4, xgb_fr_c5, xgb_fr_c6))

resamp_fr <- reshape(resamps_fr$values,
                     direction = "long",
                     varying = 2:ncol(resamps_fr$values),
                     sep = "~",
                     v.names = c("Accuracy", "Kappa", "logLoss", "ROC", "Sens", "Spec"),
                     timevar = "model")

resamp_fr <- 
  resamp_fr %>%
  mutate(model = factor(model)) %>%
  mutate(model = fct_recode(model,
                            "Demo" = "1",
                            "Demo+domain_categories" = "2",
                            "Demo+domains" = "3",
                            "Demo+app_categories" = "4",
                            "Demo+apps" = "5",
                            "Demo+bert" = "6"))

#grey.colors(3)
ggplot(resamp_fr) +
  geom_boxplot(aes(y = ROC, x = fct_rev(model), fill = model)) +
  ylim(0, 1) +
  labs(x = "") +
  labs(y = "ROC-AUC") +
  coord_flip() + 
  theme_bw() +
  theme(legend.position = "none") +
  theme(text = element_text(size = 20))
# ggsave("resamp_fr_c.png", width = 7.5, height = 7)

resamps_uk <- resamples(list(xgb_uk_c1, xgb_uk_c2, xgb_uk_c3,
                             xgb_uk_c4, xgb_uk_c5, xgb_uk_c6))

resamp_uk <- reshape(resamps_uk$values,
                     direction = "long",
                     varying = 2:ncol(resamps_uk$values),
                     sep = "~",
                     v.names = c("Accuracy", "Kappa", "logLoss", "ROC", "Sens", "Spec"),
                     timevar = "model")

resamp_uk <- 
  resamp_uk %>%
  mutate(model = factor(model)) %>%
  mutate(model = fct_recode(model,
                            "Demo" = "1",
                            "Demo+domain_categories" = "2",
                            "Demo+domains" = "3",
                            "Demo+app_categories" = "4",
                            "Demo+apps" = "5",
                            "Demo+bert" = "6"))

#grey.colors(3)
ggplot(resamp_uk) +
  geom_boxplot(aes(y = ROC, x = fct_rev(model), fill = model)) +
  ylim(0, 1) +
  labs(x = "") +
  labs(y = "ROC-AUC") +
  coord_flip() + 
  theme_bw() +
  theme(legend.position = "none") +
  theme(text = element_text(size = 20))
# ggsave("resamp_fr_c.png", width = 7.5, height = 7)

##################################################################################
# Predict in test data
##################################################################################

p_xgb_de_c1 <- predict(xgb_de_c1, newdata = track_dom_de_test, type = "prob")
p_xgb_de_c2 <- predict(xgb_de_c2, newdata = track_dom_de_test, type = "prob")
p_xgb_de_c3 <- predict(xgb_de_c3, newdata = track_dom_de_test, type = "prob")
p_xgb_de_c4 <- predict(xgb_de_c4, newdata = track_app_de_test, type = "prob")
p_xgb_de_c5 <- predict(xgb_de_c5, newdata = track_app_de_test, type = "prob")
p_xgb_de_c6 <- predict(xgb_de_c6, newdata = track_bert_de_test, type = "prob")

p_xgb_fr_c1 <- predict(xgb_fr_c1, newdata = track_dom_fr_test, type = "prob")
p_xgb_fr_c2 <- predict(xgb_fr_c2, newdata = track_dom_fr_test, type = "prob")
p_xgb_fr_c3 <- predict(xgb_fr_c3, newdata = track_dom_fr_test, type = "prob")
p_xgb_fr_c4 <- predict(xgb_fr_c4, newdata = track_app_fr_test, type = "prob")
p_xgb_fr_c5 <- predict(xgb_fr_c5, newdata = track_app_fr_test, type = "prob")
p_xgb_fr_c6 <- predict(xgb_fr_c6, newdata = track_bert_fr_test, type = "prob")

p_xgb_uk_c1 <- predict(xgb_uk_c1, newdata = track_dom_uk_test, type = "prob")
p_xgb_uk_c2 <- predict(xgb_uk_c2, newdata = track_dom_uk_test, type = "prob")
p_xgb_uk_c3 <- predict(xgb_uk_c3, newdata = track_dom_uk_test, type = "prob")
p_xgb_uk_c4 <- predict(xgb_uk_c4, newdata = track_app_uk_test, type = "prob")
p_xgb_uk_c5 <- predict(xgb_uk_c5, newdata = track_app_uk_test, type = "prob")
p_xgb_uk_c6 <- predict(xgb_uk_c6, newdata = track_bert_uk_test, type = "prob")

roc_xgb_de_c1 <- roc(response = track_dom_de_test$d_change, predictor = p_xgb_de_c1$changed)
roc_xgb_de_c2 <- roc(response = track_dom_de_test$d_change, predictor = p_xgb_de_c2$changed)
roc_xgb_de_c3 <- roc(response = track_dom_de_test$d_change, predictor = p_xgb_de_c3$changed)
roc_xgb_de_c4 <- roc(response = track_app_de_test$d_change, predictor = p_xgb_de_c4$changed)
roc_xgb_de_c5 <- roc(response = track_app_de_test$d_change, predictor = p_xgb_de_c5$changed)
roc_xgb_de_c6 <- roc(response = track_bert_de_test$d_change, predictor = p_xgb_de_c6$changed)

roc_xgb_fr_c1 <- roc(response = track_dom_fr_test$d_change, predictor = p_xgb_fr_c1$changed)
roc_xgb_fr_c2 <- roc(response = track_dom_fr_test$d_change, predictor = p_xgb_fr_c2$changed)
roc_xgb_fr_c3 <- roc(response = track_dom_fr_test$d_change, predictor = p_xgb_fr_c3$changed)
roc_xgb_fr_c4 <- roc(response = track_app_fr_test$d_change, predictor = p_xgb_fr_c4$changed)
roc_xgb_fr_c5 <- roc(response = track_app_fr_test$d_change, predictor = p_xgb_fr_c5$changed)
roc_xgb_fr_c6 <- roc(response = track_bert_fr_test$d_change, predictor = p_xgb_fr_c6$changed)

roc_xgb_uk_c1 <- roc(response = track_dom_uk_test$d_change, predictor = p_xgb_uk_c1$changed)
roc_xgb_uk_c2 <- roc(response = track_dom_uk_test$d_change, predictor = p_xgb_uk_c2$changed)
roc_xgb_uk_c3 <- roc(response = track_dom_uk_test$d_change, predictor = p_xgb_uk_c3$changed)
roc_xgb_uk_c4 <- roc(response = track_app_uk_test$d_change, predictor = p_xgb_uk_c4$changed)
roc_xgb_uk_c5 <- roc(response = track_app_uk_test$d_change, predictor = p_xgb_uk_c5$changed)
roc_xgb_uk_c6 <- roc(response = track_bert_uk_test$d_change, predictor = p_xgb_uk_c6$changed)

ggroc(list("Demo" = roc_xgb_de_c1, "Demo+domain_categories" = roc_xgb_de_c2, "Demo+domains" = roc_xgb_de_c3)) +
  geom_abline(aes(intercept = 1, slope = 1)) +
  theme(text = element_text(size = 13))
# ggsave("roc_de_c.png", width = 7.5, height = 6)

ggroc(list("Demo" = roc_xgb_fr_c1, "Demo+domain_categories" = roc_xgb_fr_c2, "Demo+domains" = roc_xgb_fr_c3)) +
  geom_abline(aes(intercept = 1, slope = 1)) +
  theme(text = element_text(size = 13))
# ggsave("roc_fr_c.png", width = 7.5, height = 6)

ggroc(list("Demo" = roc_xgb_uk_c1, "Demo+domain_categories" = roc_xgb_uk_c2, "Demo+domains" = roc_xgb_uk_c3)) +
  geom_abline(aes(intercept = 1, slope = 1)) +
  theme(text = element_text(size = 13))
# ggsave("roc_uk_c.png", width = 7.5, height = 6)

save.image("./data/work/c_results.RDATA")
