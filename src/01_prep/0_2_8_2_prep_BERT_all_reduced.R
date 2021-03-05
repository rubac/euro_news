setwd("/home/wrszemsrgyercqh/an_work/EU/")

library(tidyverse)
library(lubridate)
library(FactoMineR)
library(factoextra)
library(stringr)
library(distances)
library(e1071)
library(reshape2)
library(xtable)
source("./src/01_prep/0_2_6_1_helper_fun_topic_extraction.R")

###### Manage title encodings
### load encodings
url_bert_reduced <- read_csv2("./data/work/url_bert_new.csv")
url_bert_reduced <- mutate_at(url_bert_reduced, vars(contains("X")), as.numeric)
url_bert_reduced <- as.data.frame(url_bert_reduced)

### Create unique url ID
url_bert_reduced$url_id <- as.numeric(factor(url_bert_reduced$url, levels = unique(url_bert_reduced$url)))

### keep only one url hit by respondent
### to prevent overweighting articles manically read many many times by the same person
url_bert_reduced <- url_bert_reduced[which(!duplicated(url_bert_reduced[,c("pseudonym","url")])),]

### filter out respondents for whom we don't have much info
url_bert_reduced$count <- 1
individual.counts <- aggregate(url_bert_reduced$count, by=list(url_bert_reduced$pseudonym), FUN=sum)
summary(individual.counts)
# url_bert_reduced <- url_bert_reduced[which(url_bert_reduced$pseudonym %in% individual.counts$Group.1[which(individual.counts$x>10)]),]

### filter out articles whose titles have not been encoded
url_bert_reduced <- drop_na(url_bert_reduced, contains("X"))
url_bert_reduced <- drop_na(url_bert_reduced, pseudonym)

### Need to make sure we only have articles read before election
url_bert_reduced <- url_bert_reduced %>% 
  mutate(dt = as_date(used_at)) %>%
  filter(dt <= "2019-05-26")

encodings <- url_bert_reduced[which(!duplicated(url_bert_reduced$url_id)), which(str_detect(names(url_bert_reduced),"^X"))]
l2norm <- sqrt(rowSums(encodings^2))
encodings2 <- encodings/l2norm # normalized encodings

####################
### PCA on BERT encodings
### determine optimal number of components on unique url level

pca_res <- PCA(encodings, ncp = 50)
get_eigenvalue(pca_res)
fviz_screeplot(pca_res, ncp = 50)

### Compute factor scores and merge back to respondent level

dim <- as.data.frame(pca_res$ind$coord)[,1:6]
dim$url_id <- url_bert_reduced$url_id[which(!duplicated(url_bert_reduced$url_id))]
dim <- url_bert_reduced %>% left_join(dim, by = "url_id") %>% select(Dim.1:Dim.6, url_id, pseudonym)

dim1 <- interpret(dim, 1, url_bert_reduced)
dim1[,2] <- as.character(dim1[,2])
dim1[,2] <- str_trunc(dim1[,2], 50) 
tab <- xtable(dim1[,2:3], digits = 3) 
print(tab, type = "latex", file = "dim1.tex", include.rownames = FALSE)

dim2 <- interpret(dim, 2, url_bert_reduced)
dim2[,2] <- as.character(dim2[,2])
dim2[,2] <- str_trunc(dim2[,2], 50) 
tab <- xtable(dim2[,2:3], digits = 3) 
print(tab, type = "latex", file = "dim2.tex", include.rownames = FALSE)

dim3 <- interpret(dim, 3, url_bert_reduced)
dim3[,2] <- as.character(dim3[,2])
dim3[,2] <- str_trunc(dim3[,2], 50) 
tab <- xtable(dim3[,2:3], digits = 3) 
print(tab, type = "latex", file = "dim3.tex", include.rownames = FALSE)

dim4 <- interpret(dim, 4, url_bert_reduced)
dim4[,2] <- as.character(dim4[,2])
dim4[,2] <- str_trunc(dim4[,2], 50) 
tab <- xtable(dim4[,2:3], digits = 3) 
print(tab, type = "latex", file = "dim4.tex", include.rownames = FALSE)

dim5 <- interpret(dim, 5, url_bert_reduced)
dim5[,2] <- as.character(dim5[,2])
dim5[,2] <- str_trunc(dim5[,2], 50) 
tab <- xtable(dim5[,2:3], digits = 3) 
print(tab, type = "latex", file = "dim5.tex", include.rownames = FALSE)

dim6 <- interpret(dim, 6, url_bert_reduced)
dim6[,2] <- as.character(dim6[,2])
dim6[,2] <- str_trunc(dim6[,2], 50) 
tab <- xtable(dim6[,2:3], digits = 3) 
print(tab, type = "latex", file = "dim6.tex", include.rownames = FALSE)

### aggregate on respondent level

AC.pca <- dim %>% 
  group_by(pseudonym) %>% 
  dplyr::summarise(
    mean_dim1_r = mean(Dim.1, na.rm = TRUE),
    median_dim1_r = median(Dim.1, na.rm = TRUE),
    var_dim1_r = var(Dim.1, na.rm = TRUE),
    mean_dim2_r = mean(Dim.2, na.rm = TRUE),
    median_dim2_r = median(Dim.2, na.rm = TRUE),
    var_dim2_r = var(Dim.2, na.rm = TRUE),
    mean_dim3_r = mean(Dim.3, na.rm = TRUE),
    median_dim3_r = median(Dim.3, na.rm = TRUE),
    var_dim3_r = var(Dim.3, na.rm = TRUE),
    mean_dim4_r = mean(Dim.4, na.rm = TRUE),
    median_dim4_r = median(Dim.4, na.rm = TRUE),
    var_dim4_r = var(Dim.4, na.rm = TRUE),
    mean_dim5_r = mean(Dim.5, na.rm = TRUE),
    median_dim5_r = median(Dim.5, na.rm = TRUE),
    var_dim5_r = var(Dim.5, na.rm = TRUE),
    mean_dim6_r = mean(Dim.6, na.rm = TRUE),
    median_dim6_r = median(Dim.6, na.rm = TRUE),
    var_dim6_r = var(Dim.6, na.rm = TRUE))

AC.pca[is.na(AC.pca)] <- 0

####################
### Cluster on BERT encodings
### determine optimal number of clusters on unique url level

wss <- 2:50
for (i in 2:50) { 
  wss[i-1] <- sum(kmeans(encodings2, centers=i)$withinss)
  print(i)
}
plot(2:50, wss, type="b", xlab="Number of Clusters", ylab="Within sum of squares")

### run kmeans and merge back to respondent level

set.seed(5657)
nbclust <- 26 # 30

kmeans_res <- kmeans(encodings2, nbclust, nstart = 20, iter.max = 20)
centers <- kmeans_res$centers

url_bert_unique <- url_bert_reduced[which(!duplicated(url_bert_reduced$url_id)),]
tyti <- find.typical.titles(encodings2, centers, z = 20, url_bert_unique)
View(tyti)

tyti2 <- data.frame()
for (i in seq(1, ncol(tyti), 2)){
  tyti[,i] <- as.character(tyti[,i])
  uniques <- tyti[!duplicated(str_to_lower(tyti[,i])),c(i, i+1)] # Drop duplicates
  tyti2 <- uniques[10:20,] # Get top 10
  tyti2[,1] <- str_trunc(tyti2[,1], 50) # Truncate
  tab <- xtable(tyti2, digits = 3) # Save and print
  print(tab, type = "latex", file = paste0("cluster", (i+1)/2, ".tex"), include.rownames = FALSE)
}

ac <- data.frame(clusters = as.factor(kmeans_res$cluster))
ac$url_id <- url_bert_reduced$url_id[which(!duplicated(url_bert_reduced$url_id))]
ac$distance <- sqrt(rowSums(encodings2 - fitted(kmeans_res)) ^ 2)
ac <- url_bert_reduced %>% select(url_id, pseudonym, url, dt, active_seconds, title_en) %>% left_join(ac, by = "url_id")

### aggregate on respondent level

ac$count <- 1

AC.c0 <- ac %>% 
  group_by(pseudonym) %>% 
  dplyr::summarise(
    d_total = sum(active_seconds, na.rm = TRUE))

AC.c <- ac %>% 
  group_by(pseudonym, clusters) %>% 
  dplyr::summarise(
    n_per_clust = n(),
    d_per_clust = sum(active_seconds, na.rm = TRUE))

AC.c2 <- AC.c %>%
  left_join(AC.c0, by = "pseudonym") %>%
  dplyr::mutate(
    r_per_clust = d_per_clust / d_total) %>%
  select(-d_total, -d_per_clust)

AC.n <- spread(AC.c, clusters, n_per_clust)
AC.d <- spread(AC.c, clusters, d_per_clust)
AC.r <- spread(AC.c2, clusters, r_per_clust)

AC.cn <- AC.n %>%
  group_by(pseudonym) %>% 
  dplyr::summarise(
    clus1_n_r = sum(`1`, na.rm = TRUE),
    clus2_n_r = sum(`2`, na.rm = TRUE),
    clus3_n_r = sum(`3`, na.rm = TRUE),
    clus4_n_r = sum(`4`, na.rm = TRUE),
    clus5_n_r = sum(`5`, na.rm = TRUE),
    clus6_n_r = sum(`6`, na.rm = TRUE),
    clus7_n_r = sum(`7`, na.rm = TRUE),
    clus8_n_r = sum(`8`, na.rm = TRUE),
    clus9_n_r = sum(`9`, na.rm = TRUE),
    clus10_n_r = sum(`10`, na.rm = TRUE),
    clus11_n_r = sum(`11`, na.rm = TRUE),
    clus12_n_r = sum(`12`, na.rm = TRUE),
    clus13_n_r = sum(`13`, na.rm = TRUE),
    clus14_n_r = sum(`14`, na.rm = TRUE),
    clus15_n_r = sum(`15`, na.rm = TRUE),
    clus16_n_r = sum(`16`, na.rm = TRUE),
    clus17_n_r = sum(`17`, na.rm = TRUE),
    clus18_n_r = sum(`18`, na.rm = TRUE),
    clus19_n_r = sum(`19`, na.rm = TRUE),
    clus20_n_r = sum(`20`, na.rm = TRUE),
    clus21_n_r = sum(`21`, na.rm = TRUE),
    clus22_n_r = sum(`22`, na.rm = TRUE),
    clus23_n_r = sum(`23`, na.rm = TRUE),
    clus24_n_r = sum(`24`, na.rm = TRUE),
    clus25_n_r = sum(`25`, na.rm = TRUE),
    clus26_n_r = sum(`26`, na.rm = TRUE))

AC.cd <- AC.d %>%
  group_by(pseudonym) %>% 
  summarise(
    clus1_d_r = sum(`1`, na.rm = TRUE),
    clus2_d_r = sum(`2`, na.rm = TRUE),
    clus3_d_r = sum(`3`, na.rm = TRUE),
    clus4_d_r = sum(`4`, na.rm = TRUE),
    clus5_d_r = sum(`5`, na.rm = TRUE),
    clus6_d_r = sum(`6`, na.rm = TRUE),
    clus7_d_r = sum(`7`, na.rm = TRUE),
    clus8_d_r = sum(`8`, na.rm = TRUE),
    clus9_d_r = sum(`9`, na.rm = TRUE),
    clus10_d_r = sum(`10`, na.rm = TRUE),
    clus11_d_r = sum(`11`, na.rm = TRUE),
    clus12_d_r = sum(`12`, na.rm = TRUE),
    clus13_d_r = sum(`13`, na.rm = TRUE),
    clus14_d_r = sum(`14`, na.rm = TRUE),
    clus15_d_r = sum(`15`, na.rm = TRUE),
    clus16_d_r = sum(`16`, na.rm = TRUE),
    clus17_d_r = sum(`17`, na.rm = TRUE),
    clus18_d_r = sum(`18`, na.rm = TRUE),
    clus19_d_r = sum(`19`, na.rm = TRUE),
    clus20_d_r = sum(`20`, na.rm = TRUE),
    clus21_d_r = sum(`21`, na.rm = TRUE),
    clus22_d_r = sum(`22`, na.rm = TRUE),
    clus23_d_r = sum(`23`, na.rm = TRUE),
    clus24_d_r = sum(`24`, na.rm = TRUE),
    clus25_d_r = sum(`25`, na.rm = TRUE),
    clus26_d_r = sum(`26`, na.rm = TRUE))

AC.cr <- AC.r %>%
  group_by(pseudonym) %>% 
  summarise(
    clus1_r_r = sum(`1`, na.rm = TRUE),
    clus2_r_r = sum(`2`, na.rm = TRUE),
    clus3_r_r = sum(`3`, na.rm = TRUE),
    clus4_r_r = sum(`4`, na.rm = TRUE),
    clus5_r_r = sum(`5`, na.rm = TRUE),
    clus6_r_r = sum(`6`, na.rm = TRUE),
    clus7_r_r = sum(`7`, na.rm = TRUE),
    clus8_r_r = sum(`8`, na.rm = TRUE),
    clus9_r_r = sum(`9`, na.rm = TRUE),
    clus10_r_r = sum(`10`, na.rm = TRUE),
    clus11_r_r = sum(`11`, na.rm = TRUE),
    clus12_r_r = sum(`12`, na.rm = TRUE),
    clus13_r_r = sum(`13`, na.rm = TRUE),
    clus14_r_r = sum(`14`, na.rm = TRUE),
    clus15_r_r = sum(`15`, na.rm = TRUE),
    clus16_r_r = sum(`16`, na.rm = TRUE),
    clus17_r_r = sum(`17`, na.rm = TRUE),
    clus18_r_r = sum(`18`, na.rm = TRUE),
    clus19_r_r = sum(`19`, na.rm = TRUE),
    clus20_r_r = sum(`20`, na.rm = TRUE),
    clus21_r_r = sum(`21`, na.rm = TRUE),
    clus22_r_r = sum(`22`, na.rm = TRUE),
    clus23_r_r = sum(`23`, na.rm = TRUE),
    clus24_r_r = sum(`24`, na.rm = TRUE),
    clus25_r_r = sum(`25`, na.rm = TRUE),
    clus26_r_r = sum(`26`, na.rm = TRUE))

####################
### Merge and save

bert <- 
  AC.pca %>%
  left_join(AC.cd, by = "pseudonym") %>%
  left_join(AC.cn, by = "pseudonym") %>%
  left_join(AC.cr, by = "pseudonym")

saveRDS(bert, "./data/work/bert_all_reduced.RDS")
save(pca_res, kmeans_res, file = "./data/work/PCA_cluster.Rdata")
