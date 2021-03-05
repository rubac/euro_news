setwd("/home/wrszemsrgyercqh/an_work/EU/")

library(tidyverse)
library(lubridate)
library(FactoMineR)
library(factoextra)
library(stringr)
library(distances)
library(reshape2)
library(data.table)
source("./src/01_prep/0_2_6_1_helper_fun_topic_extraction.R")

###### Manage title encodings
### load encodings

bert_de <- fread("./data/work/title_encodings_DE.csv",stringsAsFactors = FALSE)
bert_de <- as.data.frame(bert_de)
bert_de <- rename(bert_de, title = filtred_title)

bert_fr <- fread("./data/work/title_encodings_FR.csv",stringsAsFactors = FALSE)
bert_fr <- as.data.frame(bert_fr)
bert_fr <- rename(bert_fr, title = title_fr)

load("./data/work/url_UK.RData")
bert_uk <- read.csv2("./data/work/title_encodings_UK.csv",stringsAsFactors = FALSE)
bert_uk <- url_UK %>%
  select(url, web_visits_id, id, panelist_id, used_at, active_seconds, domain, category) %>%
  right_join(bert_uk, by = "url") %>%
  mutate(v_1 = "UK",
         title_en = title_filtered,
         used_at = as.character(used_at)) %>%
  mutate_at(vars(matches("X")), as.numeric) %>%
  select(url:category, v_1, title, title_en, X0:X767)

url_bert <- bind_rows(bert_de, bert_fr, bert_uk)

### delete duplicates if any
url_bert <- url_bert[which(!duplicated(url_bert)),]

### Create unique url ID
url_bert$url_id <- as.numeric(factor(url_bert$url, levels = unique(url_bert$url)))

### keep only one url hit by respondent
### to prevent overweighting articles manically read many many times by the same person
url_bert <- url_bert[which(!duplicated(url_bert[,c("panelist_id", "url")])),]

### filter out respondents for whom we don't have much info
url_bert$count <- 1
individual.counts <- aggregate(url_bert$count, by=list(url_bert$panelist_id), FUN=sum)
summary(individual.counts)
# url_bert <- url_bert[which(url_bert$pseudonym %in% individual.counts$Group.1[which(individual.counts$x>10)]),]

### filter out articles whose titles have not been encoded
url_bert <- na.omit(url_bert)

### Need to make sure we only have articles read before election
url_bert <- url_bert %>% 
  mutate(dt = as_date(used_at)) %>%
  filter(dt <= "2019-05-26") %>%
  rename(pseudonym = panelist_id)

####################
### PCA on BERT encodings
### determine optimal number of components on unique url level

encodings <- url_bert[which(!duplicated(url_bert$url_id)), which(str_detect(names(url_bert),"^X"))]
res <- PCA(encodings, ncp = 20)
get_eigenvalue(res)
fviz_screeplot(res, ncp = 20)

### Compute factor scores and merge back to respondent level

dim <- as.data.frame(res$ind$coord)[,1:4]
dim$url_id <- url_bert$url_id[which(!duplicated(url_bert$url_id))]
dim <- url_bert %>% left_join(dim, by = "url_id") %>% select(Dim.1:Dim.4, url_id, pseudonym)

interpret(dim, 1, url_bert)
interpret(dim, 2, url_bert)
interpret(dim, 3, url_bert)
interpret(dim, 4, url_bert)

### aggregate on respondent level

AC.pca <- dim %>% 
  group_by(pseudonym) %>% 
  dplyr::summarise(
    mean_dim1 = mean(Dim.1, na.rm = TRUE),
    median_dim1 = median(Dim.1, na.rm = TRUE),
    var_dim1 = var(Dim.1, na.rm = TRUE),
    mean_dim2 = mean(Dim.2, na.rm = TRUE),
    median_dim2 = median(Dim.2, na.rm = TRUE),
    var_dim2 = var(Dim.2, na.rm = TRUE),
    mean_dim3 = mean(Dim.3, na.rm = TRUE),
    median_dim3 = median(Dim.3, na.rm = TRUE),
    var_dim3 = var(Dim.3, na.rm = TRUE),
    mean_dim4 = mean(Dim.4, na.rm = TRUE),
    median_dim4 = median(Dim.4, na.rm = TRUE),
    var_dim4 = var(Dim.4, na.rm = TRUE))

AC.pca[is.na(AC.pca)] <- 0

####################
### Cluster on BERT encodings
### determine optimal number of clusters on unique url level

wss <- 2:50
for (i in 2:50) { 
  wss[i-1] <- sum(kmeans(encodings, centers=i)$withinss)
  print(i)
}
plot(2:50, wss, type="b", xlab="Number of Clusters", ylab="Within sum of squares")

### run kmeans and merge back to respondent level

nbclust <- 45
kmeans_res <- kmeans(encodings, nbclust, nstart = 10, iter.max = 20)
centers <- kmeans_res$centers

url_bert_unique <- url_bert[which(!duplicated(url_bert$url_id)),]
tyti <- find.typical.titles(encodings, centers, z = 20, url_bert_unique)
View(tyti)

ac <- data.frame(clusters = as.factor(kmeans_res$cluster))
ac$url_id <- url_bert$url_id[which(!duplicated(url_bert$url_id))]
ac <- url_bert %>% select(url_id, pseudonym, url, dt, active_seconds) %>% left_join(ac, by = "url_id")

### aggregate on respondent level

ac$count <- 1

AC.c <- ac %>% 
  group_by(pseudonym, clusters) %>% 
  dplyr::summarise(
    n_per_clust = n(),
    d_per_clust = sum(active_seconds, na.rm = TRUE))

AC.n <- spread(AC.c, clusters, n_per_clust)
AC.d <- spread(AC.c, clusters, d_per_clust)

AC.cn <- AC.n %>%
  group_by(pseudonym) %>% 
  dplyr::summarise(
    clus1_n = sum(`1`, na.rm = TRUE),
    clus2_n = sum(`2`, na.rm = TRUE),
    clus3_n = sum(`3`, na.rm = TRUE),
    clus4_n = sum(`4`, na.rm = TRUE),
    clus5_n = sum(`5`, na.rm = TRUE),
    clus6_n = sum(`6`, na.rm = TRUE),
    clus7_n = sum(`7`, na.rm = TRUE),
    clus8_n = sum(`8`, na.rm = TRUE),
    clus9_n = sum(`9`, na.rm = TRUE),
    clus10_n = sum(`10`, na.rm = TRUE),
    clus11_n = sum(`11`, na.rm = TRUE),
    clus12_n = sum(`12`, na.rm = TRUE),
    clus13_n = sum(`13`, na.rm = TRUE),
    clus14_n = sum(`14`, na.rm = TRUE),
    clus15_n = sum(`15`, na.rm = TRUE),
    clus16_n = sum(`16`, na.rm = TRUE),
    clus17_n = sum(`17`, na.rm = TRUE),
    clus18_n = sum(`18`, na.rm = TRUE),
    clus19_n = sum(`19`, na.rm = TRUE),
    clus20_n = sum(`20`, na.rm = TRUE),
    clus21_n = sum(`21`, na.rm = TRUE),
    clus22_n = sum(`22`, na.rm = TRUE),
    clus23_n = sum(`23`, na.rm = TRUE),
    clus24_n = sum(`24`, na.rm = TRUE),
    clus25_n = sum(`25`, na.rm = TRUE),
    clus26_n = sum(`26`, na.rm = TRUE),
    clus27_n = sum(`27`, na.rm = TRUE),
    clus28_n = sum(`28`, na.rm = TRUE),
    clus29_n = sum(`29`, na.rm = TRUE),
    clus30_n = sum(`30`, na.rm = TRUE),
    clus31_n = sum(`31`, na.rm = TRUE),
    clus32_n = sum(`32`, na.rm = TRUE),
    clus33_n = sum(`33`, na.rm = TRUE),
    clus34_n = sum(`34`, na.rm = TRUE),
    clus35_n = sum(`35`, na.rm = TRUE),
    clus36_n = sum(`36`, na.rm = TRUE),
    clus37_n = sum(`37`, na.rm = TRUE),
    clus38_n = sum(`38`, na.rm = TRUE),
    clus39_n = sum(`39`, na.rm = TRUE),
    clus40_n = sum(`40`, na.rm = TRUE),
    clus41_n = sum(`41`, na.rm = TRUE),
    clus42_n = sum(`42`, na.rm = TRUE),
    clus43_n = sum(`43`, na.rm = TRUE),
    clus44_n = sum(`44`, na.rm = TRUE),
    clus45_n = sum(`45`, na.rm = TRUE))

AC.cd <- AC.d %>%
  group_by(pseudonym) %>% 
  summarise(
    clus1_d = sum(`1`, na.rm = TRUE),
    clus2_d = sum(`2`, na.rm = TRUE),
    clus3_d = sum(`3`, na.rm = TRUE),
    clus4_d = sum(`4`, na.rm = TRUE),
    clus5_d = sum(`5`, na.rm = TRUE),
    clus6_d = sum(`6`, na.rm = TRUE),
    clus7_d = sum(`7`, na.rm = TRUE),
    clus8_d = sum(`8`, na.rm = TRUE),
    clus9_d = sum(`9`, na.rm = TRUE),
    clus10_d = sum(`10`, na.rm = TRUE),
    clus11_d = sum(`11`, na.rm = TRUE),
    clus12_d = sum(`12`, na.rm = TRUE),
    clus13_d = sum(`13`, na.rm = TRUE),
    clus14_d = sum(`14`, na.rm = TRUE),
    clus15_d = sum(`15`, na.rm = TRUE),
    clus16_d = sum(`16`, na.rm = TRUE),
    clus17_d = sum(`17`, na.rm = TRUE),
    clus18_d = sum(`18`, na.rm = TRUE),
    clus19_d = sum(`19`, na.rm = TRUE),
    clus20_d = sum(`20`, na.rm = TRUE),
    clus21_d = sum(`21`, na.rm = TRUE),
    clus22_d = sum(`22`, na.rm = TRUE),
    clus23_d = sum(`23`, na.rm = TRUE),
    clus24_d = sum(`24`, na.rm = TRUE),
    clus25_d = sum(`25`, na.rm = TRUE),
    clus26_d = sum(`26`, na.rm = TRUE),
    clus27_d = sum(`27`, na.rm = TRUE),
    clus28_d = sum(`28`, na.rm = TRUE),
    clus29_d = sum(`29`, na.rm = TRUE),
    clus30_d = sum(`30`, na.rm = TRUE),
    clus31_d = sum(`31`, na.rm = TRUE),
    clus32_d = sum(`32`, na.rm = TRUE),
    clus33_d = sum(`33`, na.rm = TRUE),
    clus34_d = sum(`34`, na.rm = TRUE),
    clus35_d = sum(`35`, na.rm = TRUE),
    clus36_d = sum(`36`, na.rm = TRUE),
    clus37_d = sum(`37`, na.rm = TRUE),
    clus38_d = sum(`38`, na.rm = TRUE),
    clus39_d = sum(`39`, na.rm = TRUE),
    clus40_d = sum(`40`, na.rm = TRUE),
    clus41_d = sum(`41`, na.rm = TRUE),
    clus42_d = sum(`42`, na.rm = TRUE),
    clus43_d = sum(`43`, na.rm = TRUE),
    clus44_d = sum(`44`, na.rm = TRUE),
    clus45_d = sum(`45`, na.rm = TRUE))

####################
### Merge and save

bert <- 
  AC.pca %>%
  left_join(AC.cd, by = "pseudonym") %>%
  left_join(AC.cn, by = "pseudonym")

saveRDS(bert, "./data/work/bert_all.RDS")
