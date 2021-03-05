##################################################################################

library(tidyverse)

# Set path
setwd("/home/r_uma_2019/respondi_eu/")

##################################################################################
# Prepare data
##################################################################################

# load predictors data
# cat app data  --- reduced to app respondents
predictors.cat.app.uk <- readRDS("./data/work/pred_track_cat_app_uk.RDS")
predictors.cat.app.fr <- readRDS("./data/work/pred_track_cat_app_fr.RDS")
predictors.cat.app.de <- readRDS("./data/work/pred_track_cat_app_de.RDS")

# cat domain data
predictors.cat.dom.uk <- readRDS("./data/work/pred_track_cat_dom_uk.RDS")
predictors.cat.dom.fr <- readRDS("./data/work/pred_track_cat_dom_fr.RDS")
predictors.cat.dom.de <- readRDS("./data/work/pred_track_cat_dom_de.RDS")

# domain data only
predictors.dom.uk <- readRDS(file="./data/work/pred_track_dom_uk.RDS")
predictors.dom.fr <- readRDS(file="./data/work/pred_track_dom_fr.RDS")
predictors.dom.de <- readRDS(file="./data/work/pred_track_dom_de.RDS")

# app data only  --- reduced to app respondents
predictors.app.uk <- readRDS(file="./data/work/pred_track_app_uk.RDS")
predictors.app.fr <- readRDS(file="./data/work/pred_track_app_fr.RDS")
predictors.app.de <- readRDS(file="./data/work/pred_track_app_de.RDS")

# bert data only  --- reduced set of respondents
predictors.bert.uk <- readRDS(file="./data/work/pred_track_bert_uk.RDS")
predictors.bert.fr <- readRDS(file="./data/work/pred_track_bert_fr.RDS")
predictors.bert.de <- readRDS(file="./data/work/pred_track_bert_de.RDS")

# load survey data
sdata <- readRDS("./data/work/dat_surv.RDS")

# load background data
load("./data/orig/panel_bis.RData")

# Combine survey and background data 
spdata <- 
  panel_bis %>%
  select(-country) %>%
  rename(panelist_id = pseudonym) %>%
  left_join(sdata, by = "panelist_id")
rm(sdata, panel_bis)

# left join because we dont have socdem for all survey Respondents

# Outcomes

spdata$d_change <- as.factor(spdata$change)
spdata$d_change[spdata$change %in% c("did not vote", "doesnt remember")] <- NA
spdata$d_change <- droplevels(spdata$d_change)
levels(spdata$d_change) <- c("changed", "not_changed")

spdata$undecided <- as.factor(spdata$undecided)
spdata$leftmidright <- as.factor(spdata$leftmidright)
spdata$polinterest <- as.factor(spdata$polinterest)

spdata$undecided <- droplevels(spdata$undecided)
levels(spdata$undecided) <- c("decided", "undecided")

spdata$polinterest <- droplevels(spdata$polinterest)
levels(spdata$polinterest) <- c("low", "high")

spdata$leftmidright <- droplevels(spdata$leftmidright)
levels(spdata$leftmidright) <- c("right", "middle", "left")

table(spdata$voted)
table(spdata$change)
table(spdata$d_change)
table(spdata$undecided)
table(spdata$polinterest)
table(spdata$leftmidright)

spdata <- spdata %>% 
  filter(!is.na(voted),
         !is.na(change),
         !is.na(undecided),
         !is.na(polinterest),
         !is.na(leftmidright))

# Fill all missings in categorial columns with "NONE"
spdata[,c("gender", "age_class", "children", 
          "region_FR", "region_UK", "region_DE",
          "income_FR_DE", "income_UK", 
          "education_DE", "education_FR", "education_UK",
          "family_UK_DE", "family_FR")] <-
  lapply(spdata[,c("gender", "age_class", "children", 
                   "region_FR", "region_UK", "region_DE",
                   "income_FR_DE", "income_UK", 
                   "education_DE", "education_FR", "education_UK",
                   "family_UK_DE", "family_FR")],
         fct_explicit_na)

# Clean factor levels
for(i in c("gender", "age_class", "children", 
           "region_FR", "region_UK", "region_DE",
           "income_FR_DE", "income_UK", 
           "education_DE", "education_FR", "education_UK",
           "family_UK_DE", "family_FR")){
  levels(spdata[,i]) <- gsub("[^a-zA-Z0-9]", "",levels(spdata[,i]))
}

# Split by country
spdata_de <- 
  spdata %>%
  filter(country == "Germany") %>%
  rename(region = region_DE, income = income_FR_DE, education = education_DE, family = family_UK_DE) %>%
  select(-c(region_FR, region_UK, income_UK, education_FR, education_UK, social_class_FR, family_FR)) %>%
  droplevels()

spdata_fr <- 
  spdata %>%
  filter(country == "France") %>%
  rename(region = region_FR, income = income_FR_DE, education = education_FR, family = family_FR) %>%
  select(-c(region_UK, region_DE, income_UK, education_DE, education_UK, social_class_FR, family_UK_DE)) %>%
  droplevels()

spdata_uk <- 
  spdata %>%
  filter(country == "UK") %>%
  rename(region = region_UK, income = income_UK, education = education_UK, family = family_UK_DE) %>%
  select(-c(region_FR, region_DE, income_FR_DE, education_DE, education_FR, social_class_FR, family_FR)) %>%
  droplevels()

# Blocks of features
# Socio-demo
demo <- c("gender", "age_class", "children", "region", "income", "education", "family")

# Categories app
for (i in c("de","fr","uk")) {
  temp <- readRDS(paste0("./data/work/pred_track_cat_app_", i, ".RDS"))
  temp <- select(temp, -panelist_id)
  temp <- gsub("[^a-zA-Z0-9]", "", names(temp))
  assign(paste("pred_track_cat_app_", i, sep = ""), temp) 
}

# Categories domain
for (i in c("de","fr","uk")) {
  temp <- readRDS(paste0("./data/work/pred_track_cat_dom_", i, ".RDS"))
  temp <- select(temp, -panelist_id)
  temp <- gsub("[^a-zA-Z0-9]", "", names(temp))
  assign(paste("pred_track_cat_dom_", i, sep = ""), temp) 
}

# Domains
for (i in c("de","fr","uk")) {
  temp <- readRDS(paste0("./data/work/pred_track_dom_", i, ".RDS"))
  temp <- select(temp, -panelist_id)
  temp <- gsub("[^a-zA-Z0-9]", "", names(temp))
  assign(paste("pred_track_dom_", i, sep = ""), temp) 
}

# Apps
for (i in c("de","fr","uk")) {
  temp <- readRDS(paste0("./data/work/pred_track_app_", i, ".RDS"))
  temp <- select(temp, -panelist_id)
  temp <- gsub("[^a-zA-Z0-9]", "", names(temp))
  assign(paste("pred_track_app_", i, sep = ""), temp) 
}

# BERT
for (i in c("de","fr","uk")) {
  temp <- readRDS(paste0("./data/work/pred_track_bert_", i, ".RDS"))
  temp <- select(temp, -panelist_id)
  temp <- gsub("[^a-zA-Z0-9]", "", names(temp))
  assign(paste("pred_track_bert_", i, sep = ""), temp) 
}

# Clean tracking data

# domain data only
names(predictors.dom.de) <- gsub("[^a-zA-Z0-9]", "", names(predictors.dom.de))
predictors.dom.de <- predictors.dom.de[, !duplicated(colnames(predictors.dom.de))]

names(predictors.dom.fr) <- gsub("[^a-zA-Z0-9]", "", names(predictors.dom.fr))
predictors.dom.fr <- predictors.dom.fr[, !duplicated(colnames(predictors.dom.fr))]

names(predictors.dom.uk) <- gsub("[^a-zA-Z0-9]", "", names(predictors.dom.uk))
predictors.dom.uk <- predictors.dom.uk[, !duplicated(colnames(predictors.dom.uk))]

# app data only
names(predictors.app.de) <- gsub("[^a-zA-Z0-9]", "", names(predictors.app.de))
predictors.app.de <- predictors.app.de[, !duplicated(colnames(predictors.app.de))]

names(predictors.app.fr) <- gsub("[^a-zA-Z0-9]", "", names(predictors.app.fr))
predictors.app.fr <- predictors.app.fr[, !duplicated(colnames(predictors.app.fr))]

names(predictors.app.uk) <- gsub("[^a-zA-Z0-9]", "", names(predictors.app.uk))
predictors.app.uk <- predictors.app.uk[, !duplicated(colnames(predictors.app.uk))]

# domain cat 
names(predictors.cat.dom.de) <- gsub("[^a-zA-Z0-9]", "", names(predictors.cat.dom.de))
predictors.cat.dom.de <- predictors.cat.dom.de[, !duplicated(colnames(predictors.cat.dom.de))]

names(predictors.cat.dom.fr) <- gsub("[^a-zA-Z0-9]", "", names(predictors.cat.dom.fr))
predictors.cat.dom.fr <- predictors.cat.dom.fr[, !duplicated(colnames(predictors.cat.dom.fr))]

names(predictors.cat.dom.uk) <- gsub("[^a-zA-Z0-9]", "", names(predictors.cat.dom.uk))
predictors.cat.dom.uk <- predictors.cat.dom.uk[, !duplicated(colnames(predictors.cat.dom.uk))]

# app cat
names(predictors.cat.app.de) <- gsub("[^a-zA-Z0-9]", "", names(predictors.cat.app.de))
predictors.cat.app.de <- predictors.cat.app.de[, !duplicated(colnames(predictors.cat.app.de))]

names(predictors.cat.app.fr) <- gsub("[^a-zA-Z0-9]", "", names(predictors.cat.app.fr))
predictors.cat.app.fr <- predictors.cat.app.fr[, !duplicated(colnames(predictors.cat.app.fr))]

names(predictors.cat.app.uk) <- gsub("[^a-zA-Z0-9]", "", names(predictors.cat.app.uk))
predictors.cat.app.uk <- predictors.cat.app.uk[, !duplicated(colnames(predictors.cat.app.uk))]

# BERT
names(predictors.bert.de) <- gsub("[^a-zA-Z0-9]", "", names(predictors.bert.de))
predictors.bert.de <- predictors.bert.de[, !duplicated(colnames(predictors.bert.de))]

names(predictors.bert.fr) <- gsub("[^a-zA-Z0-9]", "", names(predictors.bert.fr))
predictors.bert.fr <- predictors.bert.fr[, !duplicated(colnames(predictors.bert.fr))]

names(predictors.bert.uk) <- gsub("[^a-zA-Z0-9]", "", names(predictors.bert.uk))
predictors.bert.uk <- predictors.bert.uk[, !duplicated(colnames(predictors.bert.uk))]

# Merge survey, panel and tracking data

# domain and cat data
track_dom_de <- 
  predictors.dom.de %>%
  left_join(predictors.cat.dom.de, by = "panelistid") %>%
  rename(panelist_id = panelistid) %>%
  left_join(spdata_de, by = "panelist_id")

track_dom_de <- track_dom_de %>% 
  filter(!is.na(voted),
         !is.na(change),
         !is.na(undecided),
         !is.na(polinterest),
         !is.na(leftmidright))

track_dom_fr <- 
  predictors.dom.fr %>%
  left_join(predictors.cat.dom.fr, by = "panelistid") %>%
  rename(panelist_id = panelistid) %>%
  left_join(spdata_fr, by = "panelist_id")

track_dom_fr <- track_dom_fr %>% 
  filter(!is.na(voted),
         !is.na(change),
         !is.na(undecided),
         !is.na(polinterest),
         !is.na(leftmidright))

track_dom_uk <- 
  predictors.dom.uk %>%
  left_join(predictors.cat.dom.uk, by = "panelistid") %>%
  rename(panelist_id = panelistid) %>%
  left_join(spdata_uk, by = "panelist_id")

track_dom_uk <- track_dom_uk %>% 
  filter(!is.na(voted),
         !is.na(change),
         !is.na(undecided),
         !is.na(polinterest),
         !is.na(leftmidright))

# app, domain and cat data  --- reduced set of respondents
track_app_de <- 
  predictors.app.de %>%
  left_join(predictors.cat.app.de, by = "panelistid") %>%
  rename(panelist_id = panelistid) %>%
  left_join(track_dom_de, by = "panelist_id")

names(track_app_de) <- gsub("\\.x", "", names(track_app_de))
names(track_app_de) <- gsub("\\.y", "", names(track_app_de))
track_app_de <- track_app_de[, !duplicated(colnames(track_app_de))]

track_app_de <- track_app_de %>% 
  filter(!is.na(voted),
         !is.na(change),
         !is.na(undecided),
         !is.na(polinterest),
         !is.na(leftmidright))

track_app_fr <- 
  predictors.app.fr %>%
  left_join(predictors.cat.app.fr, by = "panelistid") %>%
  rename(panelist_id = panelistid) %>%
  left_join(track_dom_fr, by = "panelist_id")

names(track_app_fr) <- gsub("\\.x", "", names(track_app_fr))
names(track_app_fr) <- gsub("\\.y", "", names(track_app_fr))
track_app_fr <- track_app_fr[, !duplicated(colnames(track_app_fr))]

track_app_fr <- track_app_fr %>% 
  filter(!is.na(voted),
         !is.na(change),
         !is.na(undecided),
         !is.na(polinterest),
         !is.na(leftmidright))

track_app_uk <- 
  predictors.app.uk %>%
  left_join(predictors.cat.app.uk, by = "panelistid") %>%
  rename(panelist_id = panelistid) %>%
  left_join(track_dom_uk, by = "panelist_id")

names(track_app_uk) <- gsub("\\.x", "", names(track_app_uk))
names(track_app_uk) <- gsub("\\.y", "", names(track_app_uk))
track_app_uk <- track_app_uk[, !duplicated(colnames(track_app_uk))]

track_app_uk <- track_app_uk %>% 
  filter(!is.na(voted),
         !is.na(change),
         !is.na(undecided),
         !is.na(polinterest),
         !is.na(leftmidright))

# domain, cat and BERT data
track_bert_de <- 
  predictors.bert.de %>%
  rename(panelist_id = panelistid) %>%
  left_join(track_dom_de, by = "panelist_id")

track_bert_de <- track_bert_de %>% 
  filter(!is.na(voted),
         !is.na(change),
         !is.na(undecided),
         !is.na(polinterest),
         !is.na(leftmidright))

track_bert_fr <- 
  predictors.bert.fr %>%
  rename(panelist_id = panelistid) %>%
  left_join(track_dom_fr, by = "panelist_id")

track_bert_fr <- track_bert_fr %>% 
  filter(!is.na(voted),
         !is.na(change),
         !is.na(undecided),
         !is.na(polinterest),
         !is.na(leftmidright))

track_bert_uk <- 
  predictors.bert.uk %>%
  rename(panelist_id = panelistid) %>%
  left_join(track_dom_uk, by = "panelist_id")

track_bert_uk <- track_bert_uk %>% 
  filter(!is.na(voted),
         !is.na(change),
         !is.na(undecided),
         !is.na(polinterest),
         !is.na(leftmidright))

##################################################################################
# Train-test splits
##################################################################################

# domain and cat data
set.seed(35743)
train_de <- sample(1:nrow(track_dom_de), 0.8*nrow(track_dom_de))
train_fr <- sample(1:nrow(track_dom_fr), 0.8*nrow(track_dom_fr))
train_uk <- sample(1:nrow(track_dom_uk), 0.8*nrow(track_dom_uk))

track_dom_de_train <- track_dom_de[train_de,]
track_dom_de_test <- track_dom_de[-train_de,]
track_dom_fr_train <- track_dom_fr[train_fr,]
track_dom_fr_test <- track_dom_fr[-train_fr,]
track_dom_uk_train <- track_dom_uk[train_uk,]
track_dom_uk_test <- track_dom_uk[-train_uk,]

# domain, app and cat data
set.seed(56841)
train_de <- sample(1:nrow(track_app_de), 0.8*nrow(track_app_de))
train_fr <- sample(1:nrow(track_app_fr), 0.8*nrow(track_app_fr))
train_uk <- sample(1:nrow(track_app_uk), 0.8*nrow(track_app_uk))

track_app_de_train <- track_app_de[train_de,]
track_app_de_test <- track_app_de[-train_de,]
track_app_fr_train <- track_app_fr[train_fr,]
track_app_fr_test <- track_app_fr[-train_fr,]
track_app_uk_train <- track_app_uk[train_uk,]
track_app_uk_test <- track_app_uk[-train_uk,]

# domain, bert and cat data
set.seed(34981)
train_de <- sample(1:nrow(track_bert_de), 0.8*nrow(track_bert_de))
train_fr <- sample(1:nrow(track_bert_fr), 0.8*nrow(track_bert_fr))
train_uk <- sample(1:nrow(track_bert_uk), 0.8*nrow(track_bert_uk))

track_bert_de_train <- track_bert_de[train_de,]
track_bert_de_test <- track_bert_de[-train_de,]
track_bert_fr_train <- track_bert_fr[train_fr,]
track_bert_fr_test <- track_bert_fr[-train_fr,]
track_bert_uk_train <- track_bert_uk[train_uk,]
track_bert_uk_test <- track_bert_uk[-train_uk,]

# save

save(track_app_de_train,track_app_de_test,
     track_dom_de_train, track_dom_de_test, 
     track_bert_de_train,track_bert_de_test,
     track_app_fr_train,track_app_fr_test,
     track_dom_fr_train, track_dom_fr_test, 
     track_bert_fr_train,track_bert_fr_test,
     track_app_uk_train,track_app_uk_test,
     track_dom_uk_train, track_dom_uk_test, 
     track_bert_uk_train,track_bert_uk_test,
     demo,
     pred_track_app_de,pred_track_dom_de,
     pred_track_cat_app_de,pred_track_cat_dom_de,
     pred_track_bert_de,
     pred_track_app_fr,pred_track_dom_fr,
     pred_track_cat_app_fr,pred_track_cat_dom_fr,
     pred_track_bert_fr,
     pred_track_app_uk,pred_track_dom_uk,
     pred_track_cat_app_uk,pred_track_cat_dom_uk,
     pred_track_bert_uk,
     file = "./data/work/prep_predict.Rdata")
