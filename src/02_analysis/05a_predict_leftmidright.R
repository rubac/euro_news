##################################################################################

library(tidyverse)
library(caret)
library(xgboost)
library(mboost)
library(pROC)
library(MLmetrics)

# Set path
setwd("/home/r_uma_2019/respondi_eu/")

##################################################################################
# Setup
##################################################################################

# load data
load("./data/work/prep_predict.Rdata")

# Caret Setup

evalStats <- function(...) c(multiClassSummary(...),
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

##################################################################################
# Models - XGBoost
##################################################################################

# left-mid-right - Germany

model_de_l1 <- paste("leftmidright ~", paste(demo, collapse="+"))
model_de_l2 <- paste(model_de_l1, paste("+"), paste(pred_track_cat_dom_de, collapse="+"))
model_de_l3 <- paste(model_de_l2, paste("+"), paste(pred_track_dom_de, collapse="+"))
model_de_l4 <- paste(model_de_l1, paste("+"), paste(pred_track_cat_app_de, collapse="+"))
model_de_l5 <- paste(model_de_l4, paste("+"), paste(pred_track_app_de, collapse="+"))
model_de_l6 <- paste(model_de_l1, paste("+"), paste(pred_track_bert_de, collapse="+"))

# left-mid-right - Germany - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_de_l1a <- train(",model_de_l1,",
                      data = track_dom_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_de_l1a$results$max_depth[which.max(xgb_de_l1a$results[,"AUC"])]
best_child <- xgb_de_l1a$results$min_child_weight[which.max(xgb_de_l1a$results[,"AUC"])]
best_gamma <- xgb_de_l1a$results$gamma[which.max(xgb_de_l1a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_l1 <- train(",model_de_l1,",
                      data = track_dom_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_de_l1, metric = "AUC")

# left-mid-right - Germany - socio_demo + domain-categories

set.seed(48284)
eval(parse(text=paste("xgb_de_l2a <- train(",model_de_l2,",
                      data = track_dom_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_de_l2a$results$max_depth[which.max(xgb_de_l2a$results[,"AUC"])]
best_child <- xgb_de_l2a$results$min_child_weight[which.max(xgb_de_l2a$results[,"AUC"])]
best_gamma <- xgb_de_l2a$results$gamma[which.max(xgb_de_l2a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_l2 <- train(",model_de_l2,",
                      data = track_dom_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_de_l2, metric = "AUC")

# left-mid-right - Germany - socio_demo + domain-categories + domains

set.seed(48284)
eval(parse(text=paste("xgb_de_l3a <- train(",model_de_l3,",
                      data = track_dom_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_de_l3a$results$max_depth[which.max(xgb_de_l3a$results[,"AUC"])]
best_child <- xgb_de_l3a$results$min_child_weight[which.max(xgb_de_l3a$results[,"AUC"])]
best_gamma <- xgb_de_l3a$results$gamma[which.max(xgb_de_l3a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_l3 <- train(",model_de_l3,",
                      data = track_dom_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_de_l3, metric = "AUC")

# left-mid-right - Germany - socio_demo + app-categories

set.seed(48284)
eval(parse(text=paste("xgb_de_l4a <- train(",model_de_l4,",
                      data = track_app_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_de_l4a$results$max_depth[which.max(xgb_de_l4a$results[,"AUC"])]
best_child <- xgb_de_l4a$results$min_child_weight[which.max(xgb_de_l4a$results[,"AUC"])]
best_gamma <- xgb_de_l4a$results$gamma[which.max(xgb_de_l4a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_l4 <- train(",model_de_l4,",
                      data = track_app_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_de_l4, metric = "AUC")

# left-mid-right - Germany - socio_demo + app-categories + apps

set.seed(48284)
eval(parse(text=paste("xgb_de_l5a <- train(",model_de_l5,",
                      data = track_app_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_de_l5a$results$max_depth[which.max(xgb_de_l5a$results[,"AUC"])]
best_child <- xgb_de_l5a$results$min_child_weight[which.max(xgb_de_l5a$results[,"AUC"])]
best_gamma <- xgb_de_l5a$results$gamma[which.max(xgb_de_l5a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_l5 <- train(",model_de_l5,",
                      data = track_app_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_de_l5, metric = "AUC")

# left-mid-right - Germany - socio_demo + bert

set.seed(48284)
eval(parse(text=paste("xgb_de_l6a <- train(",model_de_l6,",
                      data = track_bert_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_de_l6a$results$max_depth[which.max(xgb_de_l6a$results[,"AUC"])]
best_child <- xgb_de_l6a$results$min_child_weight[which.max(xgb_de_l6a$results[,"AUC"])]
best_gamma <- xgb_de_l6a$results$gamma[which.max(xgb_de_l6a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_de_l6 <- train(",model_de_l6,",
                      data = track_bert_de_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_de_l6, metric = "AUC")

##################################################################################

# left-mid-right - France

model_fr_l1 <- paste("leftmidright ~", paste(demo, collapse="+"))
model_fr_l2 <- paste(model_fr_l1, paste("+"), paste(pred_track_cat_dom_fr, collapse="+"))
model_fr_l3 <- paste(model_fr_l2, paste("+"), paste(pred_track_dom_fr, collapse="+"))
model_fr_l4 <- paste(model_fr_l1, paste("+"), paste(pred_track_cat_app_fr, collapse="+"))
model_fr_l5 <- paste(model_fr_l4, paste("+"), paste(pred_track_app_fr, collapse="+"))
model_fr_l6 <- paste(model_fr_l1, paste("+"), paste(pred_track_bert_fr, collapse="+"))

# left-mid-right - France - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_fr_l1a <- train(",model_fr_l1,",
                      data = track_dom_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_l1a$results$max_depth[which.max(xgb_fr_l1a$results[,"AUC"])]
best_child <- xgb_fr_l1a$results$min_child_weight[which.max(xgb_fr_l1a$results[,"AUC"])]
best_gamma <- xgb_fr_l1a$results$gamma[which.max(xgb_fr_l1a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_l1 <- train(",model_fr_l1,",
                      data = track_dom_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_fr_l1, metric = "AUC")

# left-mid-right - France - socio_demo + domain-categories

set.seed(48284)
eval(parse(text=paste("xgb_fr_l2a <- train(",model_fr_l2,",
                      data = track_dom_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_l2a$results$max_depth[which.max(xgb_fr_l2a$results[,"AUC"])]
best_child <- xgb_fr_l2a$results$min_child_weight[which.max(xgb_fr_l2a$results[,"AUC"])]
best_gamma <- xgb_fr_l2a$results$gamma[which.max(xgb_fr_l2a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_l2 <- train(",model_fr_l2,",
                      data = track_dom_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_fr_l2, metric = "AUC")

# left-mid-right - France - socio_demo + domain-categories + domains

set.seed(48284)
eval(parse(text=paste("xgb_fr_l3a <- train(",model_fr_l3,",
                      data = track_dom_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_l3a$results$max_depth[which.max(xgb_fr_l3a$results[,"AUC"])]
best_child <- xgb_fr_l3a$results$min_child_weight[which.max(xgb_fr_l3a$results[,"AUC"])]
best_gamma <- xgb_fr_l3a$results$gamma[which.max(xgb_fr_l3a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_l3 <- train(",model_fr_l3,",
                      data = track_dom_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_fr_l3, metric = "AUC")

# left-mid-right - France - socio_demo + app-categories

set.seed(48284)
eval(parse(text=paste("xgb_fr_l4a <- train(",model_fr_l4,",
                      data = track_app_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_l4a$results$max_depth[which.max(xgb_fr_l4a$results[,"AUC"])]
best_child <- xgb_fr_l4a$results$min_child_weight[which.max(xgb_fr_l4a$results[,"AUC"])]
best_gamma <- xgb_fr_l4a$results$gamma[which.max(xgb_fr_l4a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_l4 <- train(",model_fr_l4,",
                      data = track_app_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_fr_l4, metric = "AUC")

# left-mid-right - France - socio_demo + app-categories + apps

set.seed(48284)
eval(parse(text=paste("xgb_fr_l5a <- train(",model_fr_l5,",
                      data = track_app_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_l5a$results$max_depth[which.max(xgb_fr_l5a$results[,"AUC"])]
best_child <- xgb_fr_l5a$results$min_child_weight[which.max(xgb_fr_l5a$results[,"AUC"])]
best_gamma <- xgb_fr_l5a$results$gamma[which.max(xgb_fr_l5a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_l5 <- train(",model_fr_l5,",
                      data = track_app_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_fr_l5, metric = "AUC")

# left-mid-right - France - socio_demo + bert

set.seed(48284)
eval(parse(text=paste("xgb_fr_l6a <- train(",model_fr_l6,",
                      data = track_bert_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_fr_l6a$results$max_depth[which.max(xgb_fr_l6a$results[,"AUC"])]
best_child <- xgb_fr_l6a$results$min_child_weight[which.max(xgb_fr_l6a$results[,"AUC"])]
best_gamma <- xgb_fr_l6a$results$gamma[which.max(xgb_fr_l6a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_fr_l6 <- train(",model_fr_l6,",
                      data = track_bert_fr_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_fr_l6, metric = "AUC")

##################################################################################

# left-mid-right - UK

model_uk_l1 <- paste("leftmidright ~", paste(demo, collapse="+"))
model_uk_l2 <- paste(model_uk_l1, paste("+"), paste(pred_track_cat_dom_uk, collapse="+"))
model_uk_l3 <- paste(model_uk_l2, paste("+"), paste(pred_track_dom_uk, collapse="+"))
model_uk_l4 <- paste(model_uk_l1, paste("+"), paste(pred_track_cat_app_uk, collapse="+"))
model_uk_l5 <- paste(model_uk_l4, paste("+"), paste(pred_track_app_uk, collapse="+"))
model_uk_l6 <- paste(model_uk_l1, paste("+"), paste(pred_track_bert_uk, collapse="+"))

# left-mid-right - UK - socio_demo

set.seed(48284)
eval(parse(text=paste("xgb_uk_l1a <- train(",model_uk_l1,",
                      data = track_dom_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_l1a$results$max_depth[which.max(xgb_uk_l1a$results[,"AUC"])]
best_child <- xgb_uk_l1a$results$min_child_weight[which.max(xgb_uk_l1a$results[,"AUC"])]
best_gamma <- xgb_uk_l1a$results$gamma[which.max(xgb_uk_l1a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_l1 <- train(",model_uk_l1,",
                      data = track_dom_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_uk_l1, metric = "AUC")

# left-mid-right - UK - socio_demo + domain-categories

set.seed(48284)
eval(parse(text=paste("xgb_uk_l2a <- train(",model_uk_l2,",
                      data = track_dom_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_l2a$results$max_depth[which.max(xgb_uk_l2a$results[,"AUC"])]
best_child <- xgb_uk_l2a$results$min_child_weight[which.max(xgb_uk_l2a$results[,"AUC"])]
best_gamma <- xgb_uk_l2a$results$gamma[which.max(xgb_uk_l2a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_l2 <- train(",model_uk_l2,",
                      data = track_dom_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_uk_l2, metric = "AUC")

# left-mid-right - UK - socio_demo + domain-categories + domains

set.seed(48284)
eval(parse(text=paste("xgb_uk_l3a <- train(",model_uk_l3,",
                      data = track_dom_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_l3a$results$max_depth[which.max(xgb_uk_l3a$results[,"AUC"])]
best_child <- xgb_uk_l3a$results$min_child_weight[which.max(xgb_uk_l3a$results[,"AUC"])]
best_gamma <- xgb_uk_l3a$results$gamma[which.max(xgb_uk_l3a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_l3 <- train(",model_uk_l3,",
                      data = track_dom_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_uk_l3, metric = "AUC")

# left-mid-right - UK - socio_demo + app-categories

set.seed(48284)
eval(parse(text=paste("xgb_uk_l4a <- train(",model_uk_l4,",
                      data = track_app_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_l4a$results$max_depth[which.max(xgb_uk_l4a$results[,"AUC"])]
best_child <- xgb_uk_l4a$results$min_child_weight[which.max(xgb_uk_l4a$results[,"AUC"])]
best_gamma <- xgb_uk_l4a$results$gamma[which.max(xgb_uk_l4a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_l4 <- train(",model_uk_l4,",
                      data = track_app_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_uk_l4, metric = "AUC")

# left-mid-right - UK - socio_demo + app-categories + apps

set.seed(48284)
eval(parse(text=paste("xgb_uk_l5a <- train(",model_uk_l5,",
                      data = track_app_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_l5a$results$max_depth[which.max(xgb_uk_l5a$results[,"AUC"])]
best_child <- xgb_uk_l5a$results$min_child_weight[which.max(xgb_uk_l5a$results[,"AUC"])]
best_gamma <- xgb_uk_l5a$results$gamma[which.max(xgb_uk_l5a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_l5 <- train(",model_uk_l5,",
                      data = track_app_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_uk_l5, metric = "AUC")

# left-mid-right - UK - socio_demo + bert

set.seed(48284)
eval(parse(text=paste("xgb_uk_l6a <- train(",model_uk_l6,",
                      data = track_bert_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid0,
                      metric = 'AUC',
                      na.action = na.omit)")))

best_depth <- xgb_uk_l6a$results$max_depth[which.max(xgb_uk_l6a$results[,"AUC"])]
best_child <- xgb_uk_l6a$results$min_child_weight[which.max(xgb_uk_l6a$results[,"AUC"])]
best_gamma <- xgb_uk_l6a$results$gamma[which.max(xgb_uk_l6a$results[,"AUC"])]

xgb_grid <- expand.grid(max_depth = c(best_depth-1, best_depth, best_depth+1),
                        nrounds = c(250, 500, 750, 1000),
                        eta = c(0.025, 0.01),
                        min_child_weight = best_child,
                        subsample = c(0.7, 1),
                        gamma = best_gamma,
                        colsample_bytree = c(0.7, 1))
xgb_grid <- filter(xgb_grid, max_depth >= 1)

set.seed(29483)
eval(parse(text=paste("xgb_uk_l6 <- train(",model_uk_l6,",
                      data = track_bert_uk_train,
                      method = 'xgbTree',
                      trControl = ctrl,
                      tuneGrid = xgb_grid,
                      metric = 'AUC',
                      na.action = na.omit)")))
plot(xgb_uk_l6, metric = "AUC")

##################################################################################
# Variable Importance
##################################################################################

plot(varImp(xgb_de_l1), top = 10)
plot(varImp(xgb_de_l2), top = 10)
plot(varImp(xgb_de_l3), top = 10)
plot(varImp(xgb_de_l4), top = 10)
plot(varImp(xgb_de_l5), top = 10)
plot(varImp(xgb_de_l6), top = 10)

varImp(xgb_de_l6)$importance %>% 
  as.data.frame() %>%
  rownames_to_column() %>%
  arrange(Overall) %>%
  top_n(15, Overall) %>% 
  mutate(rowname = forcats::fct_inorder(rowname )) %>%
  ggplot() +
  geom_col(aes(x = rowname, y = Overall)) +
  labs(y = "Importance", x = "") +
  coord_flip()
# ggsave("imp_de_l.png", width = 6, height = 6)

plot(varImp(xgb_fr_l1), top = 10)
plot(varImp(xgb_fr_l2), top = 10)
plot(varImp(xgb_fr_l3), top = 10)
plot(varImp(xgb_fr_l4), top = 10)
plot(varImp(xgb_fr_l5), top = 10)
plot(varImp(xgb_fr_l6), top = 10)

varImp(xgb_fr_l6)$importance %>% 
  as.data.frame() %>%
  rownames_to_column() %>%
  arrange(Overall) %>%
  top_n(15, Overall) %>% 
  mutate(rowname = forcats::fct_inorder(rowname )) %>%
  ggplot() +
  geom_col(aes(x = rowname, y = Overall)) +
  labs(y = "Importance", x = "") +
  coord_flip()
# ggsave("imp_fr_l.png", width = 6, height = 6)

plot(varImp(xgb_uk_l1), top = 10)
plot(varImp(xgb_uk_l2), top = 10)
plot(varImp(xgb_uk_l3), top = 10)
plot(varImp(xgb_uk_l4), top = 10)
plot(varImp(xgb_uk_l5), top = 10)
plot(varImp(xgb_uk_l6), top = 10)

varImp(xgb_uk_l6)$importance %>% 
  as.data.frame() %>%
  rownames_to_column() %>%
  arrange(Overall) %>%
  top_n(15, Overall) %>% 
  mutate(rowname = forcats::fct_inorder(rowname )) %>%
  ggplot() +
  geom_col(aes(x = rowname, y = Overall)) +
  labs(y = "Importance", x = "") +
  coord_flip()
# ggsave("imp_uk_l.png", width = 6, height = 6)

##################################################################################
# PDPs
##################################################################################

##################################################################################
# Compare CV performance
##################################################################################

resamps_de <- resamples(list(xgb_de_l1, xgb_de_l2, xgb_de_l3,
                             xgb_de_l4, xgb_de_l5, xgb_de_l6))

resamp_de <- reshape(resamps_de$values,
                     direction = "long",
                     varying = 2:ncol(resamps_de$values),
                     sep = "~",
                     v.names = c("Accuracy", "AUC", "Kappa", "logLoss", "logLoss.1", 
                                 "Mean_Balanced_Accuracy", "Mean_Detection_Rate", "Mean_F1",
                                 "Mean_Neg_Pred_Value", "Mean_Pos_Pred_Value", "Mean_Precision",
                                 "Mean_Recall", "Mean_Sensitivity", "Mean_Specificity", "prAUC"),
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
  geom_boxplot(aes(y = AUC, x = fct_rev(model), fill = model)) +
  ylim(0, 1) +
  labs(x = "") +
  labs(y = "ROC-AUC") +
  coord_flip() + 
  theme_bw() +
  theme(legend.position = "none") +
  theme(text = element_text(size = 20))
# ggsave("resamp_de_l.png", width = 7.5, height = 7)

resamps_fr <- resamples(list(xgb_fr_l1, xgb_fr_l2, xgb_fr_l3,
                             xgb_fr_l4, xgb_fr_l5, xgb_fr_l6))

resamp_fr <- reshape(resamps_fr$values,
                     direction = "long",
                     varying = 2:ncol(resamps_fr$values),
                     sep = "~",
                     v.names = c("Accuracy", "AUC", "Kappa", "logLoss", "logLoss.1", 
                                 "Mean_Balanced_Accuracy", "Mean_Detection_Rate", "Mean_F1",
                                 "Mean_Neg_Pred_Value", "Mean_Pos_Pred_Value", "Mean_Precision",
                                 "Mean_Recall", "Mean_Sensitivity", "Mean_Specificity", "prAUC"),
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
  geom_boxplot(aes(y = AUC, x = fct_rev(model), fill = model)) +
  ylim(0, 1) +
  labs(x = "") +
  labs(y = "ROC-AUC") +
  coord_flip() + 
  theme_bw() +
  theme(legend.position = "none") +
  theme(text = element_text(size = 20))
# ggsave("resamp_fr_l.png", width = 7.5, height = 7)

resamps_uk <- resamples(list(xgb_uk_l1, xgb_uk_l2, xgb_uk_l3,
                             xgb_uk_l4, xgb_uk_l5, xgb_uk_l6))

resamp_uk <- reshape(resamps_uk$values,
                     direction = "long",
                     varying = 2:ncol(resamps_uk$values),
                     sep = "~",
                     v.names = c("Accuracy", "AUC", "Kappa", "logLoss", "logLoss.1", 
                                 "Mean_Balanced_Accuracy", "Mean_Detection_Rate", "Mean_F1",
                                 "Mean_Neg_Pred_Value", "Mean_Pos_Pred_Value", "Mean_Precision",
                                 "Mean_Recall", "Mean_Sensitivity", "Mean_Specificity", "prAUC"),
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
  geom_boxplot(aes(y = AUC, x = fct_rev(model), fill = model)) +
  ylim(0, 1) +
  labs(x = "") +
  labs(y = "ROC-AUC") +
  coord_flip() + 
  theme_bw() +
  theme(legend.position = "none") +
  theme(text = element_text(size = 20))
# ggsave("resamp_fr_l.png", width = 7.5, height = 7)

##################################################################################
# Predict in test data
##################################################################################

p_xgb_de_l1 <- predict(xgb_de_l1, newdata = track_dom_de_test, type = "prob")
p_xgb_de_l2 <- predict(xgb_de_l2, newdata = track_dom_de_test, type = "prob")
p_xgb_de_l3 <- predict(xgb_de_l3, newdata = track_dom_de_test, type = "prob")
p_xgb_de_l4 <- predict(xgb_de_l4, newdata = track_app_de_test, type = "prob")
p_xgb_de_l5 <- predict(xgb_de_l5, newdata = track_app_de_test, type = "prob")
p_xgb_de_l6 <- predict(xgb_de_l6, newdata = track_bert_de_test, type = "prob")

p_xgb_fr_l1 <- predict(xgb_fr_l1, newdata = track_dom_fr_test, type = "prob")
p_xgb_fr_l2 <- predict(xgb_fr_l2, newdata = track_dom_fr_test, type = "prob")
p_xgb_fr_l3 <- predict(xgb_fr_l3, newdata = track_dom_fr_test, type = "prob")
p_xgb_fr_l4 <- predict(xgb_fr_l4, newdata = track_app_fr_test, type = "prob")
p_xgb_fr_l5 <- predict(xgb_fr_l5, newdata = track_app_fr_test, type = "prob")
p_xgb_fr_l6 <- predict(xgb_fr_l6, newdata = track_bert_fr_test, type = "prob")

p_xgb_uk_l1 <- predict(xgb_uk_l1, newdata = track_dom_uk_test, type = "prob")
p_xgb_uk_l2 <- predict(xgb_uk_l2, newdata = track_dom_uk_test, type = "prob")
p_xgb_uk_l3 <- predict(xgb_uk_l3, newdata = track_dom_uk_test, type = "prob")
p_xgb_uk_l4 <- predict(xgb_uk_l4, newdata = track_app_uk_test, type = "prob")
p_xgb_uk_l5 <- predict(xgb_uk_l5, newdata = track_app_uk_test, type = "prob")
p_xgb_uk_l6 <- predict(xgb_uk_l6, newdata = track_bert_uk_test, type = "prob")

#roc_xgb_de_l1 <- roc(response = track_dom_de_test$polinterest, predictor = p_xgb_de_l1$low)
#roc_xgb_de_l2 <- roc(response = track_dom_de_test$polinterest, predictor = p_xgb_de_l2$low)
#roc_xgb_de_l3 <- roc(response = track_dom_de_test$polinterest, predictor = p_xgb_de_l3$low)
#roc_xgb_de_l4 <- roc(response = track_app_de_test$polinterest, predictor = p_xgb_de_l4$low)
#roc_xgb_de_l5 <- roc(response = track_app_de_test$polinterest, predictor = p_xgb_de_l5$low)
#roc_xgb_de_l6 <- roc(response = track_bert_de_test$polinterest, predictor = p_xgb_de_l6$low)

#roc_xgb_fr_l1 <- roc(response = track_dom_fr_test$polinterest, predictor = p_xgb_fr_l1$low)
#roc_xgb_fr_l2 <- roc(response = track_dom_fr_test$polinterest, predictor = p_xgb_fr_l2$low)
#roc_xgb_fr_l3 <- roc(response = track_dom_fr_test$polinterest, predictor = p_xgb_fr_l3$low)
#roc_xgb_fr_l4 <- roc(response = track_app_fr_test$polinterest, predictor = p_xgb_fr_l4$low)
#roc_xgb_fr_l5 <- roc(response = track_app_fr_test$polinterest, predictor = p_xgb_fr_l5$low)
#roc_xgb_fr_l6 <- roc(response = track_bert_fr_test$polinterest, predictor = p_xgb_fr_l6$low)

#roc_xgb_uk_l1 <- roc(response = track_dom_uk_test$polinterest, predictor = p_xgb_uk_l1$low)
#roc_xgb_uk_l2 <- roc(response = track_dom_uk_test$polinterest, predictor = p_xgb_uk_l2$low)
#roc_xgb_uk_l3 <- roc(response = track_dom_uk_test$polinterest, predictor = p_xgb_uk_l3$low)
#roc_xgb_uk_l4 <- roc(response = track_app_uk_test$polinterest, predictor = p_xgb_uk_l4$low)
#roc_xgb_uk_l5 <- roc(response = track_app_uk_test$polinterest, predictor = p_xgb_uk_l5$low)
#roc_xgb_uk_l6 <- roc(response = track_bert_uk_test$polinterest, predictor = p_xgb_uk_l6$low)

#ggroc(list("Demo" = roc_xgb_de_l1, "Demo+domain_categories" = roc_xgb_de_l2, "Demo+domains" = roc_xgb_de_l3)) +
#  geom_abline(aes(intercept = 1, slope = 1)) +
#  theme(text = element_text(size = 13))
# ggsave("roc_de_l.png", width = 7.5, height = 6)

#ggroc(list("Demo" = roc_xgb_fr_l1, "Demo+domain_categories" = roc_xgb_fr_l2, "Demo+domains" = roc_xgb_fr_l3)) +
#  geom_abline(aes(intercept = 1, slope = 1)) +
#  theme(text = element_text(size = 13))
# ggsave("roc_fr_l.png", width = 7.5, height = 6)

#ggroc(list("Demo" = roc_xgb_uk_l1, "Demo+domain_categories" = roc_xgb_uk_l2, "Demo+domains" = roc_xgb_uk_l3)) +
#  geom_abline(aes(intercept = 1, slope = 1)) +
#  theme(text = element_text(size = 13))
# ggsave("roc_uk_l.png", width = 7.5, height = 6)

save.image("./data/work/l_results.RDATA")
