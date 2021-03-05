##################################################################################

library(tidyverse)
library(caret)
library(broom)
library(dotwhisker)
library(pROC)
library(rtf)

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

ctrl  <- trainControl(method = "repeatedcv",
                      number = 10,
                      repeats = 5,
                      summaryFunction = evalStats,
                      classProbs = TRUE,
                      verboseIter = TRUE)

# glmnet grid

grid <- expand.grid(alpha = c(0, 0.5, 1),
                    lambda = seq(1, 0, length = 50))

bert_pca_cluster_reduced_small <- c("clus10_n_r", "clus16_n_r", "clus23_n_r",
                                    "median_dim2_r", "var_dim2_r")

bert_pca_cluster_reduced_s_de <- c("clus23_r_r", "median_dim2_r", "var_dim2_r")
bert_pca_cluster_reduced_s_fr <- c("clus16_r_r", "median_dim2_r", "var_dim2_r")
bert_pca_cluster_reduced_s_uk <- c("clus10_r_r", "median_dim2_r", "var_dim2_r")

model_de_v1 <- paste("undecided ~", paste(demo_de, collapse="+"))
model_de_v2 <- paste("undecided ~", paste(demo_de, collapse="+"),
                     paste("+"), paste(pred_track_cat_dom_all, collapse="+"), 
                     paste("+"), paste(pred_track_cat_app_all, collapse="+"))
model_de_v3 <- paste("undecided ~", paste(demo_de, collapse="+"), 
                     paste("+"), paste(pred_track_dom_all, collapse="+"), 
                     paste("+"), paste(pred_track_app_all, collapse="+"), paste("+ no_app_data"))
model_de_v4 <- paste("undecided ~", paste(demo_de, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_pca, collapse="+"), 
                     paste("+"), paste(bert_pca_cluster_reduced_cn, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_cr, collapse="+"), paste("+ no_reduced_bert"))
model_de_v5 <- paste("undecided ~", paste(demo_de, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_small, collapse="+"), paste("+ no_reduced_bert"))
model_de_v6 <- paste("undecided ~", paste(demo_de, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_s_de, collapse="+"), paste("+ no_reduced_bert"))

model_fr_v1 <- paste("undecided ~", paste(demo_fr, collapse="+"))
model_fr_v2 <- paste("undecided ~", paste(demo_fr, collapse="+"),
                     paste("+"), paste(pred_track_cat_dom_all, collapse="+"), 
                     paste("+"), paste(pred_track_cat_app_all, collapse="+"))
model_fr_v3 <- paste("undecided ~", paste(demo_fr, collapse="+"),
                     paste("+"), paste(pred_track_dom_all, collapse="+"), 
                     paste("+"), paste(pred_track_app_all, collapse="+"), paste("+ no_app_data"))
model_fr_v4 <- paste("undecided ~", paste(demo_fr, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_pca, collapse="+"), 
                     paste("+"), paste(bert_pca_cluster_reduced_cn, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_cr, collapse="+"), paste("+ no_reduced_bert"))
model_fr_v5 <- paste("undecided ~", paste(demo_fr, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_small, collapse="+"), paste("+ no_reduced_bert"))
model_fr_v6 <- paste("undecided ~", paste(demo_fr, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_s_fr, collapse="+"), paste("+ no_reduced_bert"))

model_uk_v1 <- paste("undecided ~", paste(demo_uk, collapse="+"))
model_uk_v2 <- paste("undecided ~", paste(demo_uk, collapse="+"),
                     paste("+"), paste(pred_track_cat_dom_all, collapse="+"), 
                     paste("+"), paste(pred_track_cat_app_all, collapse="+"))
model_uk_v3 <- paste("undecided ~", paste(demo_uk, collapse="+"),
                     paste("+"), paste(pred_track_dom_all, collapse="+"), 
                     paste("+"), paste(pred_track_app_all, collapse="+"), paste("+ no_app_data"))
model_uk_v4 <- paste("undecided ~", paste(demo_uk, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_pca, collapse="+"), 
                     paste("+"), paste(bert_pca_cluster_reduced_cn, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_cr, collapse="+"), paste("+ no_reduced_bert"))
model_uk_v5 <- paste("undecided ~", paste(demo_uk, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_small, collapse="+"), paste("+ no_reduced_bert"))
model_uk_v5 <- paste("undecided ~", paste(demo_uk, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_small, collapse="+"), paste("+ no_reduced_bert"))
model_uk_v6 <- paste("undecided ~", paste(demo_uk, collapse="+"),
                     paste("+"), paste(bert_pca_cluster_reduced_s_uk, collapse="+"), paste("+ no_reduced_bert"))

##################################################################################
# Models - Logit + glmnet
##################################################################################

# undecided - Germany - socio_demo

set.seed(84831)
eval(parse(text=paste("glmnet_de_v1 <- train(",model_de_v1,",
                      data = track_de,
                      method = 'glmnet',
                      family = 'binomial',
                      trControl = ctrl,
                      tuneGrid = grid, 
                      metric = 'ROC',
                      na.action = na.omit)")))

# undecided - Germany - socio_demo + categories

set.seed(84831)
eval(parse(text=paste("glmnet_de_v2 <- train(",model_de_v2,",
                      data = track_de,
                      method = 'glmnet',
                      family = 'binomial',
                      trControl = ctrl,
                      tuneGrid = grid, 
                      metric = 'ROC',
                      na.action = na.omit)")))

# undecided - Germany - socio_demo + domains/apps

set.seed(84831)
eval(parse(text=paste("glmnet_de_v3 <- train(",model_de_v3,",
                      data = track_de,
                      method = 'glmnet',
                      family = 'binomial',
                      trControl = ctrl,
                      tuneGrid = grid, 
                      metric = 'ROC',
                      na.action = na.omit)")))

# undecided - Germany - socio_demo + reduced BERT

set.seed(84831)
eval(parse(text=paste("glmnet_de_v4 <- train(",model_de_v4,",
                      data = track_de,
                      method = 'glmnet',
                      family = 'binomial',
                      trControl = ctrl,
                      tuneGrid = grid, 
                      metric = 'ROC',
                      na.action = na.omit)")))

set.seed(84831)
eval(parse(text=paste("glm_de_v5 <- train(",model_de_v5,",
                      data = track_de,
                      method = 'glm',
                      family = 'binomial',
                      trControl = ctrl,
                      na.action = na.omit)")))

set.seed(84831)
eval(parse(text=paste("glm_de_v6 <- train(",model_de_v6,",
                      data = track_de,
                      method = 'glm',
                      family = 'binomial',
                      trControl = ctrl,
                      na.action = na.omit)")))

##################################################################################

# undecided - France - socio_demo

set.seed(84831)
eval(parse(text=paste("glmnet_fr_v1 <- train(",model_fr_v1,",
                      data = track_fr,
                      method = 'glmnet',
                      family = 'binomial',
                      trControl = ctrl,
                      tuneGrid = grid, 
                      metric = 'ROC',
                      na.action = na.omit)")))

# undecided - France - socio_demo + categories

set.seed(84831)
eval(parse(text=paste("glmnet_fr_v2 <- train(",model_fr_v2,",
                      data = track_fr,
                      method = 'glmnet',
                      family = 'binomial',
                      trControl = ctrl,
                      tuneGrid = grid, 
                      metric = 'ROC',
                      na.action = na.omit)")))

# undecided - France - socio_demo + domains/apps

set.seed(84831)
eval(parse(text=paste("glmnet_fr_v3 <- train(",model_fr_v3,",
                      data = track_fr,
                      method = 'glmnet',
                      family = 'binomial',
                      trControl = ctrl,
                      tuneGrid = grid, 
                      metric = 'ROC',
                      na.action = na.omit)")))

# undecided - France - socio_demo + reduced BERT

set.seed(84831)
eval(parse(text=paste("glmnet_fr_v4 <- train(",model_fr_v4,",
                      data = track_fr,
                      method = 'glmnet',
                      family = 'binomial',
                      trControl = ctrl,
                      tuneGrid = grid, 
                      metric = 'ROC',
                      na.action = na.omit)")))

set.seed(84831)
eval(parse(text=paste("glm_fr_v5 <- train(",model_fr_v5,",
                      data = track_fr,
                      method = 'glm',
                      family = 'binomial',
                      trControl = ctrl,
                      na.action = na.omit)")))

set.seed(84831)
eval(parse(text=paste("glm_fr_v6 <- train(",model_fr_v6,",
                      data = track_fr,
                      method = 'glm',
                      family = 'binomial',
                      trControl = ctrl,
                      na.action = na.omit)")))

##################################################################################

# undecided - UK - socio_demo

set.seed(84831)
eval(parse(text=paste("glmnet_uk_v1 <- train(",model_uk_v1,",
                      data = track_uk,
                      method = 'glmnet',
                      family = 'binomial',
                      trControl = ctrl,
                      tuneGrid = grid, 
                      metric = 'ROC',
                      na.action = na.omit)")))

# undecided - UK - socio_demo + categories

set.seed(84831)
eval(parse(text=paste("glmnet_uk_v2 <- train(",model_uk_v2,",
                      data = track_uk,
                      method = 'glmnet',
                      family = 'binomial',
                      trControl = ctrl,
                      tuneGrid = grid, 
                      metric = 'ROC',
                      na.action = na.omit)")))

# undecided - UK - socio_demo + domains/apps

set.seed(84831)
eval(parse(text=paste("glmnet_uk_v3 <- train(",model_uk_v3,",
                      data = track_uk,
                      method = 'glmnet',
                      family = 'binomial',
                      trControl = ctrl,
                      tuneGrid = grid, 
                      metric = 'ROC',
                      na.action = na.omit)")))

# undecided - UK - socio_demo + reduced BERT

set.seed(84831)
eval(parse(text=paste("glmnet_uk_v4 <- train(",model_uk_v4,",
                      data = track_uk,
                      method = 'glmnet',
                      family = 'binomial',
                      trControl = ctrl,
                      tuneGrid = grid, 
                      metric = 'ROC',
                      na.action = na.omit)")))

set.seed(84831)
eval(parse(text=paste("glm_uk_v5 <- train(",model_uk_v5,",
                      data = track_uk,
                      method = 'glm',
                      family = 'binomial',
                      trControl = ctrl,
                      na.action = na.omit)")))

set.seed(84831)
eval(parse(text=paste("glm_uk_v6 <- train(",model_uk_v6,",
                      data = track_uk,
                      method = 'glm',
                      family = 'binomial',
                      trControl = ctrl,
                      na.action = na.omit)")))

save(glmnet_de_v1, glmnet_de_v2, glmnet_de_v3, glmnet_de_v4, glm_de_v5, glm_de_v6,
     glmnet_fr_v1, glmnet_fr_v2, glmnet_fr_v3, glmnet_fr_v4, glm_fr_v5, glm_fr_v6,
     glmnet_uk_v1, glmnet_uk_v2, glmnet_uk_v3, glmnet_uk_v4, glm_uk_v5, glm_uk_v6,
     file = "u_results_all3.Rdata")

##################################################################################
# Logit coefplots
##################################################################################

glm_de <- tidy(glm_de_v5$finalModel) %>% 
  by_2sd(track_de) %>% 
  filter(term %in% c("clus10_n_r", "clus16_n_r", "clus23_n_r", 
                     "median_dim2_r", "var_dim2_r")) %>% 
  relabel_predictors(clus10_n_r = "Brexit Cluster",
                     clus16_n_r = "French Politics Cluster",
                     clus23_n_r = "Elections Cluster",
                     median_dim2_r = "Crime vs. Elections (Median)",
                     var_dim2_r = "Crime vs. Elections (Var)") %>% 
  mutate(model = "Germany")

glm_fr <- tidy(glm_fr_v5$finalModel) %>% 
  by_2sd(track_fr) %>% 
  filter(term %in% c("clus10_n_r", "clus16_n_r", "clus23_n_r", 
                     "median_dim2_r", "var_dim2_r")) %>% 
  relabel_predictors(clus10_n_r = "Brexit Cluster",
                     clus16_n_r = "French Politics Cluster",
                     clus23_n_r = "Elections Cluster",
                     median_dim2_r = "Crime vs. Elections (Median)",
                     var_dim2_r = "Crime vs. Elections (Var)") %>% 
  mutate(model = "France")

glm_uk <- tidy(glm_uk_v5$finalModel) %>% 
  by_2sd(track_uk) %>% 
  filter(term %in% c("clus10_n_r", "clus16_n_r", "clus23_n_r", 
                     "median_dim2_r", "var_dim2_r")) %>% 
  relabel_predictors(clus10_n_r = "Brexit Cluster",
                     clus16_n_r = "French Politics Cluster",
                     clus23_n_r = "Elections Cluster",
                     median_dim2_r = "Crime vs. Elections (Median)",
                     var_dim2_r = "Crime vs. Elections (Var)") %>% 
  mutate(model = "UK")

models <- rbind(glm_de, glm_fr, glm_uk)

dwplot(models, 
       vline = geom_vline(xintercept = 0, colour = "grey50", linetype = 2)) +
  xlab("Coefficient") + 
  ylab("") +
  theme(legend.title = element_blank(),
        text = element_text(size = 15))

ggsave("u_coef.png", width = 7.5, height = 6)

glm_de <- tidy(glm_de_v6$finalModel) %>% 
  by_2sd(track_de) %>% 
  filter(term %in% c("clus10_r_r", "clus16_r_r", "clus23_r_r", 
                     "median_dim2_r", "var_dim2_r")) %>% 
  relabel_predictors(clus23_r_r = "Elections Cluster",
                     median_dim2_r = "Crime vs. Elections (Median)",
                     var_dim2_r = "Crime vs. Elections (Var)") %>% 
  mutate(model = "Germany")

glm_fr <- tidy(glm_fr_v6$finalModel) %>% 
  by_2sd(track_fr) %>% 
  filter(term %in% c("clus10_r_r", "clus16_r_r", "clus23_r_r", 
                     "median_dim2_r", "var_dim2_r")) %>% 
  relabel_predictors(clus16_r_r = "French Politics Cluster",
                     median_dim2_r = "Crime vs. Elections (Median)",
                     var_dim2_r = "Crime vs. Elections (Var)") %>% 
  mutate(model = "France")

glm_uk <- tidy(glm_uk_v6$finalModel) %>% 
  by_2sd(track_uk) %>% 
  filter(term %in% c("clus10_r_r", "clus16_r_r", "clus23_r_r", 
                     "median_dim2_r", "var_dim2_r")) %>% 
  relabel_predictors(clus10_r_r = "Brexit Cluster",
                     median_dim2_r = "Crime vs. Elections (Median)",
                     var_dim2_r = "Crime vs. Elections (Var)") %>% 
  mutate(model = "UK")

models <- rbind(glm_de, glm_fr, glm_uk)

models$term <- fct_relevel(models$term, 
                           "Brexit Cluster", "French Politics Cluster", "Elections Cluster",
                           "Crime vs. Elections (Median)", "Crime vs. Elections (Var)")

dwplot(models, 
       vline = geom_vline(xintercept = 0, colour = "grey50", linetype = 2)) +
  xlab("Coefficient") + 
  ylab("") +
  theme(legend.title = element_blank(),
        text = element_text(size = 17))

ggsave("u_coef2.png", width = 7.5, height = 6)

##################################################################################
# Compare CV performance
##################################################################################

resamps_de <- resamples(list(glmnet_de_v1, glmnet_de_v2, glmnet_de_v3, glmnet_de_v4))
resamps_fr <- resamples(list(glmnet_fr_v1, glmnet_fr_v2, glmnet_fr_v3, glmnet_fr_v4))
resamps_uk <- resamples(list(glmnet_uk_v1, glmnet_uk_v2, glmnet_uk_v3, glmnet_uk_v4))

resamp_de <- resamps_de$values %>%
  pivot_longer(!Resample, 
               names_to = c("Model", "Metric"), 
               names_sep = "~",
               values_to = "perf") %>%
  mutate(country = "Germany")

resamp_fr <- resamps_fr$values %>%
  pivot_longer(!Resample, 
               names_to = c("Model", "Metric"), 
               names_sep = "~",
               values_to = "perf") %>%
  mutate(country = "France")

resamp_uk <- resamps_uk$values %>%
  pivot_longer(!Resample, 
               names_to = c("Model", "Metric"), 
               names_sep = "~",
               values_to = "perf") %>%
  mutate(country = "UK")

resamp <- resamp_de %>%
  add_row(resamp_fr) %>%
  add_row(resamp_uk) %>%
  mutate(Model = fct_recode(Model,
                            "Soc-Demo." = "Model1",
                            "Categories" = "Model2",
                            "Domains/Apps" = "Model3",
                            "BERT" = "Model4"))

resamp$country <- relevel(as.factor(resamp$country), ref = "Germany")

resamp %>%
  filter(Metric == "ROC") %>%
  ggplot() +
  geom_boxplot(aes(y = perf, x = fct_rev(Model), fill = country)) +
  geom_hline(yintercept = 0.5) +
  ylim(0.25, 0.9) +
  labs(x = "") +
  labs(y = "ROC-AUC") +
  coord_flip() +
  theme(legend.title = element_blank(),
        text = element_text(size = 17))

ggsave("u_perf.png", width = 6, height = 6)

resamp %>%
  filter(Metric == "ROC") %>%
  ggplot() +
  geom_boxplot(aes(y = perf, x = country, fill = fct_rev(Model))) +
  geom_hline(yintercept = 0.5) +
  ylim(0.25, 0.9) +
  labs(x = "") +
  labs(y = "ROC-AUC") +
  coord_flip() +
  theme(legend.title = element_blank(),
        text = element_text(size = 17))

ggsave("u_perf2.png", width = 6, height = 6)
