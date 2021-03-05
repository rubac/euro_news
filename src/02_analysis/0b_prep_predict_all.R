##################################################################################

library(tidyverse)
library(rtf)

# Set path
# setwd("/home/r_uma_2019/respondi_eu/")

##################################################################################
# Prepare data
##################################################################################

# load predictors data
# cat app data  --- reduced to app respondents
predictors.cat.app.all <- readRDS("./data/work/pred_track_cat_app_all.RDS")

# cat domain data
predictors.cat.dom.all <- readRDS("./data/work/pred_track_cat_dom_all.RDS")

# domain data only
predictors.dom.all <- readRDS(file="./data/work/pred_track_dom_all.RDS")

# app data only  --- reduced to app respondents
predictors.app.all <- readRDS(file="./data/work/pred_track_app_all.RDS")

# bert data  --- reduced set of respondents
bert.all <- readRDS(file="./data/work/bert_all.RDS")

# bert reduced data  --- reduced set of respondents
bert.reduced.all <- readRDS(file="./data/work/bert_all_reduced.RDS")

# topic data  --- reduced set of respondents
topics.all <- readRDS(file="./data/work/topics_all.RDS")

# load survey data
sdata <- readRDS("./data/work/dat_surv.rds")

# load background data
socdem <- readRDS("./data/work/sociodemo.rds")

# Combine survey and background data 
spdata <- 
  sdata %>%
  right_join(socdem, by = c("panelist_id" = "pseudonym"))

# Outcomes

# 1 = Voted
spdata$voted <- relevel(spdata$voted, ref = "No") 

# 1 = Changed
spdata$changed <- as.factor(spdata$change)
spdata$changed[spdata$change %in% c("did not vote", "doesnt remember")] <- NA
spdata$changed <- droplevels(spdata$changed)
levels(spdata$changed) <- c("changed", "not_changed")
spdata$changed <- relevel(spdata$changed, ref = "not_changed") 

# 1 = Undecided
spdata$undecided <- as.factor(spdata$undecided)
spdata$undecided <- droplevels(spdata$undecided)
levels(spdata$undecided) <- c("decided", "undecided")

# 1 = High
spdata$polinterest <- as.factor(spdata$polinterest)
spdata$polinterest <- droplevels(spdata$polinterest)
levels(spdata$polinterest) <- c("low", "high")

# 1 = Left
spdata$left <- ifelse(spdata$leftmidright.num < 5, "left", "not_left")
spdata$left <- as.factor(spdata$left)
spdata$left <- relevel(spdata$left, ref = "not_left") 

# 1 = Right
spdata$right <- ifelse(spdata$leftmidright.num > 7, "right", "not_right")
spdata$right <- as.factor(spdata$right)

table(spdata$voted)
table(spdata$changed)
table(spdata$undecided)
table(spdata$polinterest)
table(spdata$left)
table(spdata$right)

# Fill all missings in categorial columns with "NONE"
spdata[,c("gender", "children", 
          "region", "region_FR", "region_UK", "region_DE",
          "income", "income_FR_DE", "income_UK",
          "education", "education_DE", "education_FR", "education_UK", 
          "family", "family_UK_DE", "family_FR",
          "hh_size", "place_of_residence", "home_owner", 
          "empl_status", "empl_status2", "empl_type", "empl_job")] <-
  lapply(spdata[,c("gender", "children", 
                   "region", "region_FR", "region_UK", "region_DE",
                   "income", "income_FR_DE", "income_UK",
                   "education", "education_DE", "education_FR", "education_UK", 
                   "family", "family_UK_DE", "family_FR", 
                   "hh_size", "place_of_residence", "home_owner", 
                   "empl_status", "empl_status2", "empl_type", "empl_job")],
         fct_explicit_na)

# Clean factor levels
for(i in c("gender", "children", 
           "region", "region_FR", "region_UK", "region_DE",
           "income", "income_FR_DE", "income_UK",
           "education", "education_DE", "education_FR", "education_UK", 
           "family", "family_UK_DE", "family_FR", 
           "hh_size", "place_of_residence", "home_owner", 
           "empl_status", "empl_status2", "empl_type", "empl_job")){
  levels(spdata[,i]) <- gsub("[^a-zA-Z0-9]", "",levels(spdata[,i]))
}

##################################################################################
# Blocks of features
##################################################################################

# Socio-demo
demo <- c("gender", "children", "region", "income", "education", 
          "family", "hh_size", "place_of_residence", "home_owner", 
          "empl_status", "empl_type", "empl_job", "country", "age")
demo_de <- c("gender", "income_FR_DE", "education_DE", 
          "family_UK_DE", "hh_size", 
          "empl_status2", "age")
demo_fr <- c("gender", "income_FR_DE", "education_FR", 
          "family_FR", "hh_size",
          "empl_status2", "age")
demo_uk <- c("gender", "income_UK", "education_UK", 
          "family_UK_DE", "hh_size",
          "empl_status2", "age")

media <- c("tv_hh", "has.twitter", "has.facebook", "has.instagram", "has.linkedin", "has.oth.smedia")

# Categories app
predictors.cat.app.all$country <- NULL
names(predictors.cat.app.all) <- gsub("[^a-zA-Z0-9]", "", names(predictors.cat.app.all))
predictors.cat.app.all <- predictors.cat.app.all[, !duplicated(colnames(predictors.cat.app.all))]
pred_track_cat_app_all <- names(select(predictors.cat.app.all, -panelistid))

# Categories domain
predictors.cat.dom.all$country <- NULL
names(predictors.cat.dom.all) <- gsub("[^a-zA-Z0-9]", "", names(predictors.cat.dom.all))
predictors.cat.dom.all <- predictors.cat.dom.all[, !duplicated(colnames(predictors.cat.dom.all))]
pred_track_cat_dom_all <- names(select(predictors.cat.dom.all, -panelistid))

# Domains
names(predictors.dom.all) <- gsub("[^a-zA-Z0-9]", "", names(predictors.dom.all))
predictors.dom.all <- predictors.dom.all[, !duplicated(colnames(predictors.dom.all))]
pred_track_dom_all <- names(select(predictors.dom.all, -panelistid))

# Apps
names(predictors.app.all) <- gsub("[^a-zA-Z0-9]", "", names(predictors.app.all))
predictors.app.all <- predictors.app.all[, !duplicated(colnames(predictors.app.all))]
pred_track_app_all <- names(select(predictors.app.all, -panelistid))

# BERT - PCA and Cluster
bert_pca_cluster_all <- names(select(bert.all, -pseudonym))

# BERT reduced - PCA and Cluster
bert_pca_cluster_reduced_full <- names(select(bert.reduced.all, -pseudonym))
bert_pca_cluster_reduced_pca <- names(select(bert.reduced.all, contains("dim")))
bert_pca_cluster_reduced_cn <- names(select(bert.reduced.all, contains("_n_r")))
bert_pca_cluster_reduced_cd <- names(select(bert.reduced.all, contains("_d_r")))
bert_pca_cluster_reduced_cr <- names(select(bert.reduced.all, contains("_r_r")))
bert_pca_cluster_reduced_all <- c(bert_pca_cluster_reduced_pca, 
                                  bert_pca_cluster_reduced_cn, 
                                  bert_pca_cluster_reduced_cd)

# Topics - LCA and NMF
topics_all <- names(select(topics.all, -pseudonym))

# Merge survey and tracking data

track_data <- 
  spdata %>%
  inner_join(predictors.cat.dom.all, by = c("panelist_id" = "panelistid")) %>% # drop cases w/o tracking data
  inner_join(predictors.dom.all, by = c("panelist_id" = "panelistid")) %>%
  left_join(predictors.cat.app.all, by = c("panelist_id" = "panelistid")) %>%
  left_join(predictors.app.all, by = c("panelist_id" = "panelistid")) %>%
  left_join(bert.all, by = c("panelist_id" = "pseudonym")) %>%
  left_join(bert.reduced.all, by = c("panelist_id" = "pseudonym")) %>%
  left_join(topics.all, by = c("panelist_id" = "pseudonym"))

# Fill in zeros for respondents without app tracking or bert data

track_data$no_app_data <- ifelse(is.na(track_data$dtotalacat), "no_apps", "apps")
track_data$no_app_data <- as.factor(track_data$no_app_data)

track_data[pred_track_cat_app_all][is.na(track_data[pred_track_cat_app_all])] <- 0
sum(is.na(track_data[,pred_track_cat_app_all]))

track_data[pred_track_app_all][is.na(track_data[pred_track_app_all])] <- 0
sum(is.na(track_data[,pred_track_app_all]))

track_data$no_bert_data <- ifelse(is.na(track_data$clus1_d), "no_bert", "bert")
track_data$no_bert_data <- as.factor(track_data$no_bert_data)

track_data[bert_pca_cluster_all][is.na(track_data[bert_pca_cluster_all])] <- 0
sum(is.na(track_data[,bert_pca_cluster_all]))

track_data$no_reduced_bert <- ifelse(is.na(track_data$clus1_d_r), "no_reduced_bert", "reduced_bert")
track_data$no_reduced_bert <- as.factor(track_data$no_reduced_bert)
track_data$no_reduced_bert <- relevel(track_data$no_reduced_bert, ref = "reduced_bert")

track_data[bert_pca_cluster_reduced_full][is.na(track_data[bert_pca_cluster_reduced_full])] <- 0
sum(is.na(track_data[,bert_pca_cluster_reduced_full]))

track_data$no_topics <- ifelse(is.na(track_data$lda1_n), "no_topics", "topics")
track_data$no_topics <- as.factor(track_data$no_topics)
track_data$no_topics <- relevel(track_data$no_topics, ref = "topics")

track_data[topics_all][is.na(track_data[topics_all])] <- 0
sum(is.na(track_data[,topics_all]))

sum(is.na(track_data[demo]))

##################################################################################
# Train-test splits
##################################################################################

set.seed(34572)

track_de <- track_data[track_data$country == "Germany",]
test_de <- sample(1:nrow(track_de), 0.5*nrow(track_de))
track_fr <- track_data[track_data$country == "France",]
test_fr <- sample(1:nrow(track_fr), 0.5*nrow(track_fr))
track_uk <- track_data[track_data$country == "UK",]
test_uk <- sample(1:nrow(track_uk), 0.5*nrow(track_uk))

track_de_train <- rbind(track_de[-test_de,], track_data[track_data$country != "Germany",])
track_de_test <- track_de[test_de,]
track_fr_train <- rbind(track_fr[-test_fr,], track_data[track_data$country != "France",])
track_fr_test <- track_fr[test_fr,]
track_uk_train <- rbind(track_uk[-test_uk,], track_data[track_data$country != "UK",])
track_uk_test <- track_uk[test_uk,]

summary(track_de_test$country)
summary(track_fr_test$country)
summary(track_uk_test$country)

track_de <- filter(track_de, empl_status2 != "Missing")
track_de <- filter(track_de, income_FR_DE != "Missing")
track_fr <- filter(track_fr, income_FR_DE != "Missing")
track_uk <- filter(track_uk, income_UK != "Missing")

track_de <- droplevels(track_de)
track_fr <- droplevels(track_fr)
track_uk <- droplevels(track_uk)

# save

save(track_de, track_fr, track_uk,
     track_de_train, track_de_test,
     track_fr_train, track_fr_test,
     track_uk_train, track_uk_test,
     demo, demo_de, demo_fr, demo_uk, 
     media,
     pred_track_app_all, pred_track_dom_all,
     pred_track_cat_app_all, pred_track_cat_dom_all,
     bert_pca_cluster_all, 
     bert_pca_cluster_reduced_all,
     bert_pca_cluster_reduced_pca,
     bert_pca_cluster_reduced_cn,
     bert_pca_cluster_reduced_cd,
     bert_pca_cluster_reduced_cr,
     topics_all,
     file = "./src/02_analysis/prep_predict_all.Rdata")
