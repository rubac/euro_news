setwd("/home/wrszemsrgyercqh/an_work/EU/")

library(tidyverse)
library(FactoMineR)
library(ggplot2)
library(stringr)
library(distances)
# library(plyr)
library(reshape2)
# library(tidyverse)
library(data.table)
source("./euro-tracking/src/01_prep/0_2_6_1_helper_fun_topic_extraction.R")

# Get clusters of news articles from BERT encodings --- DE
###### manage title encodings
### load encodings
te = fread("./data/work/title_encodings_DE.csv",stringsAsFactors = FALSE)
te <- as.data.frame(te)

# names(te) <- substring(names(te), 3)
# names(te) = substr(names(te),1,nchar(names(te))-1)

### delete duplicates if any
# te = te[which(!duplicated(te)),]

### convert vector dimensions columns to numeric
for (i in 12:ncol(te)){te[,i]=as.numeric(te[,i])}

### merge url hits with encodings
euDE = te

### keep only one url hit by respondent
### to prevent overweighting articles manically read many many times by the same person
names(euDE)[which(names(euDE)=="panelist_id")]="pseudonym"
euDE = euDE[which(!duplicated(euDE[,c("pseudonym","url")])),]

### filter out respondents for whom we don't have much info
### filter out articles whose titles have not been encoded
euDE$count = 1
individual.counts = aggregate(euDE$count,by=list(euDE$pseudonym),FUN=sum)
summary(individual.counts)
# euDE = euDE[which(euDE$pseudonym %in% individual.counts$Group.1[which(individual.counts$x>10)]),]
euDE = na.omit(euDE)

####################
### PCA on BERT encoding
### interpreting dimensions and "who reads what"
for.PCA = euDE[,which(str_detect(names(euDE),"^X"))]
res = PCA(for.PCA)
dim = as.data.frame(res$ind$coord)
dim$index = 1:nrow(dim)
dim$pseudonym = euDE$pseudonym

################# determination du nombre optimal de cluster
encodings = euDE[,which(str_detect(names(euDE),"^X"))]
wss <- 2:50
for (i in 2:50) wss[i-1] <- sum(kmeans(encodings,
                                       centers=i)$withinss)
plot(2:50, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
################# creation des clusters dim 1 et 2 aprÃ¨s normalisation

nbclust <- 50
kmeans.comment <- kmeans(encodings,nbclust)
cluster.kmeans <- as.factor(kmeans.comment$cluster)
table(cluster.kmeans)
centers = kmeans.comment$centers
nrow(centers)

ac = data.frame(clusters = cluster.kmeans,pseudonym = euDE$pseudonym)
ac$count = 1
ac$used_at <- euDE$used_at
ac$url <- euDE$url
ac$active_seconds <- euDE$active_seconds
ac$active_seconds <- as.numeric(ac$active_seconds)

AC.2 <- ac %>% 
  group_by(pseudonym, clusters) %>% 
  dplyr::summarise(
    n_per_clust = n(),
    d_per_clust = sum(active_seconds, na.rm = TRUE))

AC.n <- spread(AC.2, clusters, n_per_clust)
AC.d <- spread(AC.2, clusters, d_per_clust)


AC.n <- AC.n %>%
  group_by(pseudonym) %>% 
  dplyr::summarise(
    clus1 = sum(`1`, na.rm = TRUE),
    clus2 = sum(`2`, na.rm = TRUE),
    clus3 = sum(`3`, na.rm = TRUE),
    clus4 = sum(`4`, na.rm = TRUE),
    clus5 = sum(`5`, na.rm = TRUE),
    clus6 = sum(`6`, na.rm = TRUE),
    clus7 = sum(`7`, na.rm = TRUE),
    clus8 = sum(`8`, na.rm = TRUE),
    clus9 = sum(`9`, na.rm = TRUE),
    clus10 = sum(`10`, na.rm = TRUE),
    clus11 = sum(`11`, na.rm = TRUE),
    clus12 = sum(`12`, na.rm = TRUE),
    clus13 = sum(`13`, na.rm = TRUE),
    clus14 = sum(`14`, na.rm = TRUE),
    clus15 = sum(`15`, na.rm = TRUE),
    clus16 = sum(`16`, na.rm = TRUE),
    clus17 = sum(`17`, na.rm = TRUE),
    clus18 = sum(`18`, na.rm = TRUE),
    clus19 = sum(`19`, na.rm = TRUE),
    clus20 = sum(`20`, na.rm = TRUE),
    clus21 = sum(`21`, na.rm = TRUE),
    clus22 = sum(`22`, na.rm = TRUE),
    clus23 = sum(`23`, na.rm = TRUE),
    clus24 = sum(`24`, na.rm = TRUE),
    clus25 = sum(`25`, na.rm = TRUE),
    clus26 = sum(`26`, na.rm = TRUE),
    clus27 = sum(`27`, na.rm = TRUE),
    clus28 = sum(`28`, na.rm = TRUE),
    clus29 = sum(`29`, na.rm = TRUE),
    clus30 = sum(`30`, na.rm = TRUE),
    clus31 = sum(`31`, na.rm = TRUE),
    clus32 = sum(`32`, na.rm = TRUE),
    clus33= sum(`33`, na.rm = TRUE),
    clus34= sum(`34`, na.rm = TRUE),
    clus35= sum(`35`, na.rm = TRUE),
    clus36= sum(`36`, na.rm = TRUE),
    clus37= sum(`37`, na.rm = TRUE),
    clus38= sum(`38`, na.rm = TRUE),
    clus39= sum(`39`, na.rm = TRUE),
    clus40= sum(`40`, na.rm = TRUE),
    clus41= sum(`41`, na.rm = TRUE),
    clus42= sum(`42`, na.rm = TRUE),
    clus43= sum(`43`, na.rm = TRUE),
    clus44= sum(`44`, na.rm = TRUE),
    clus45= sum(`45`, na.rm = TRUE),
    clus46= sum(`46`, na.rm = TRUE),
    clus47= sum(`47`, na.rm = TRUE),
    clus48= sum(`48`, na.rm = TRUE),
    clus49= sum(`49`, na.rm = TRUE),
    clus50= sum(`50`, na.rm = TRUE)
  )

AC.d <- AC.d %>%
  group_by(pseudonym) %>% 
  summarise(
    clus1 = sum(`1`, na.rm = TRUE),
    clus2 = sum(`2`, na.rm = TRUE),
    clus3 = sum(`3`, na.rm = TRUE),
    clus4 = sum(`4`, na.rm = TRUE),
    clus5 = sum(`5`, na.rm = TRUE),
    clus6 = sum(`6`, na.rm = TRUE),
    clus7 = sum(`7`, na.rm = TRUE),
    clus8 = sum(`8`, na.rm = TRUE),
    clus9 = sum(`9`, na.rm = TRUE),
    clus10 = sum(`10`, na.rm = TRUE),
    clus11 = sum(`11`, na.rm = TRUE),
    clus12 = sum(`12`, na.rm = TRUE),
    clus13 = sum(`13`, na.rm = TRUE),
    clus14 = sum(`14`, na.rm = TRUE),
    clus15 = sum(`15`, na.rm = TRUE),
    clus16 = sum(`16`, na.rm = TRUE),
    clus17 = sum(`17`, na.rm = TRUE),
    clus18 = sum(`18`, na.rm = TRUE),
    clus19 = sum(`19`, na.rm = TRUE),
    clus20 = sum(`20`, na.rm = TRUE),
    clus21 = sum(`21`, na.rm = TRUE),
    clus22 = sum(`22`, na.rm = TRUE),
    clus23 = sum(`23`, na.rm = TRUE),
    clus24 = sum(`24`, na.rm = TRUE),
    clus25 = sum(`25`, na.rm = TRUE),
    clus26 = sum(`26`, na.rm = TRUE),
    clus27 = sum(`27`, na.rm = TRUE),
    clus28 = sum(`28`, na.rm = TRUE),
    clus29 = sum(`29`, na.rm = TRUE),
    clus30 = sum(`30`, na.rm = TRUE),
    clus31 = sum(`31`, na.rm = TRUE),
    clus32 = sum(`32`, na.rm = TRUE),
    clus33= sum(`33`, na.rm = TRUE),
    clus34= sum(`34`, na.rm = TRUE),
    clus35= sum(`35`, na.rm = TRUE),
    clus36= sum(`36`, na.rm = TRUE),
    clus37= sum(`37`, na.rm = TRUE),
    clus38= sum(`38`, na.rm = TRUE),
    clus39= sum(`39`, na.rm = TRUE),
    clus40= sum(`40`, na.rm = TRUE),
    clus41= sum(`41`, na.rm = TRUE),
    clus42= sum(`42`, na.rm = TRUE),
    clus43= sum(`43`, na.rm = TRUE),
    clus44= sum(`44`, na.rm = TRUE),
    clus45= sum(`45`, na.rm = TRUE),
    clus46= sum(`46`, na.rm = TRUE),
    clus47= sum(`47`, na.rm = TRUE),
    clus48= sum(`48`, na.rm = TRUE),
    clus49= sum(`49`, na.rm = TRUE),
    clus50= sum(`50`, na.rm = TRUE)
  )

saveRDS(AC.d, "./data/work/dcluster_DE.RDS")
saveRDS(AC.n, "./data/work/ncluster_DE.RDS")

rm(list = ls())

#################################################################################
#### FRANCE

###### manage title encodings
### load encodings
te = read.csv2("./data/work/title_encodings_FR.csv",stringsAsFactors = FALSE)

### delete duplicates if any
te = te[which(!duplicated(te)),]

### convert vector dimensions columns to numeric
for (i in 12:ncol(te)){te[,i]=as.numeric(te[,i])}

### merge url hits with encodings
euFR = te

### keep only one url hit by respondent
### to prevent overweighting articles manically read many many times by the same person
names(euFR)[which(names(euFR)=="panelist_id")]="pseudonym"
euFR = euFR[which(!duplicated(euFR[,c("pseudonym","url")])),]

### filter out respondents for whom we don't have much info
### filter out articles whose titles have not been encoded
euFR$count = 1
individual.counts = aggregate(euFR$count,by=list(euFR$pseudonym),FUN=sum)
summary(individual.counts)
euFR = euFR[which(euFR$pseudonym %in% individual.counts$Group.1[which(individual.counts$x>10)]),]
euFR = na.omit(euFR)

####################
### PCA on BERT encoding
### interpreting dimensions and "who reads what"
for.PCA = euFR[,which(str_detect(names(euFR),"^X"))]
res = PCA(for.PCA)
dim = as.data.frame(res$ind$coord)
dim$index = 1:nrow(dim)
dim$pseudonym = euFR$pseudonym

################# determination du nombre optimal de cluster
encodings = euFR[,which(str_detect(names(euFR),"^X"))]
wss <- 2:50
for (i in 2:50) wss[i-1] <- sum(kmeans(encodings,
                                       centers=i)$withinss)

################# creation des clusters dim 1 et 2 aprÃ¨s normalisation

nbclust <- 50
kmeans.comment <- kmeans(encodings,nbclust)
cluster.kmeans <- as.factor(kmeans.comment$cluster)
table(cluster.kmeans)
centers = kmeans.comment$centers
nrow(centers)

ac = data.frame(clusters = cluster.kmeans,pseudonym = euFR$pseudonym)
ac$count = 1
ac$used_at <- euFR$used_at
ac$url <- euFR$url
ac$active_seconds <- euFR$active_seconds
ac$active_seconds <- as.numeric(ac$active_seconds)

AC.2 <- ac %>% 
  group_by(pseudonym, clusters) %>% 
  summarise(
    n_per_clust = n(),
    d_per_clust = sum(active_seconds, na.rm = TRUE))

AC.n <- spread(AC.2, clusters, n_per_clust)
AC.d <- spread(AC.2, clusters, d_per_clust)


AC.n <- AC.n %>%
  group_by(pseudonym) %>% 
  summarise(
    clus1 = sum(`1`, na.rm = TRUE),
    clus2 = sum(`2`, na.rm = TRUE),
    clus3 = sum(`3`, na.rm = TRUE),
    clus4 = sum(`4`, na.rm = TRUE),
    clus5 = sum(`5`, na.rm = TRUE),
    clus6 = sum(`6`, na.rm = TRUE),
    clus7 = sum(`7`, na.rm = TRUE),
    clus8 = sum(`8`, na.rm = TRUE),
    clus9 = sum(`9`, na.rm = TRUE),
    clus10 = sum(`10`, na.rm = TRUE),
    clus11 = sum(`11`, na.rm = TRUE),
    clus12 = sum(`12`, na.rm = TRUE),
    clus13 = sum(`13`, na.rm = TRUE),
    clus14 = sum(`14`, na.rm = TRUE),
    clus15 = sum(`15`, na.rm = TRUE),
    clus16 = sum(`16`, na.rm = TRUE),
    clus17 = sum(`17`, na.rm = TRUE),
    clus18 = sum(`18`, na.rm = TRUE),
    clus19 = sum(`19`, na.rm = TRUE),
    clus20 = sum(`20`, na.rm = TRUE),
    clus21 = sum(`21`, na.rm = TRUE),
    clus22 = sum(`22`, na.rm = TRUE),
    clus23 = sum(`23`, na.rm = TRUE),
    clus24 = sum(`24`, na.rm = TRUE),
    clus25 = sum(`25`, na.rm = TRUE),
    clus26 = sum(`26`, na.rm = TRUE),
    clus27 = sum(`27`, na.rm = TRUE),
    clus28 = sum(`28`, na.rm = TRUE),
    clus29 = sum(`29`, na.rm = TRUE),
    clus30 = sum(`30`, na.rm = TRUE),
    clus31 = sum(`31`, na.rm = TRUE),
    clus32 = sum(`32`, na.rm = TRUE),
    clus33= sum(`33`, na.rm = TRUE),
    clus34= sum(`34`, na.rm = TRUE),
    clus35= sum(`35`, na.rm = TRUE),
    clus36= sum(`36`, na.rm = TRUE),
    clus37= sum(`37`, na.rm = TRUE),
    clus38= sum(`38`, na.rm = TRUE),
    clus39= sum(`39`, na.rm = TRUE),
    clus40= sum(`40`, na.rm = TRUE),
    clus41= sum(`41`, na.rm = TRUE),
    clus42= sum(`42`, na.rm = TRUE),
    clus43= sum(`43`, na.rm = TRUE),
    clus44= sum(`44`, na.rm = TRUE),
    clus45= sum(`45`, na.rm = TRUE),
    clus46= sum(`46`, na.rm = TRUE),
    clus47= sum(`47`, na.rm = TRUE),
    clus48= sum(`48`, na.rm = TRUE),
    clus49= sum(`49`, na.rm = TRUE),
    clus50= sum(`50`, na.rm = TRUE)
  )

AC.d <- AC.d %>%
  group_by(pseudonym) %>% 
  summarise(
    clus1 = sum(`1`, na.rm = TRUE),
    clus2 = sum(`2`, na.rm = TRUE),
    clus3 = sum(`3`, na.rm = TRUE),
    clus4 = sum(`4`, na.rm = TRUE),
    clus5 = sum(`5`, na.rm = TRUE),
    clus6 = sum(`6`, na.rm = TRUE),
    clus7 = sum(`7`, na.rm = TRUE),
    clus8 = sum(`8`, na.rm = TRUE),
    clus9 = sum(`9`, na.rm = TRUE),
    clus10 = sum(`10`, na.rm = TRUE),
    clus11 = sum(`11`, na.rm = TRUE),
    clus12 = sum(`12`, na.rm = TRUE),
    clus13 = sum(`13`, na.rm = TRUE),
    clus14 = sum(`14`, na.rm = TRUE),
    clus15 = sum(`15`, na.rm = TRUE),
    clus16 = sum(`16`, na.rm = TRUE),
    clus17 = sum(`17`, na.rm = TRUE),
    clus18 = sum(`18`, na.rm = TRUE),
    clus19 = sum(`19`, na.rm = TRUE),
    clus20 = sum(`20`, na.rm = TRUE),
    clus21 = sum(`21`, na.rm = TRUE),
    clus22 = sum(`22`, na.rm = TRUE),
    clus23 = sum(`23`, na.rm = TRUE),
    clus24 = sum(`24`, na.rm = TRUE),
    clus25 = sum(`25`, na.rm = TRUE),
    clus26 = sum(`26`, na.rm = TRUE),
    clus27 = sum(`27`, na.rm = TRUE),
    clus28 = sum(`28`, na.rm = TRUE),
    clus29 = sum(`29`, na.rm = TRUE),
    clus30 = sum(`30`, na.rm = TRUE),
    clus31 = sum(`31`, na.rm = TRUE),
    clus32 = sum(`32`, na.rm = TRUE),
    clus33= sum(`33`, na.rm = TRUE),
    clus34= sum(`34`, na.rm = TRUE),
    clus35= sum(`35`, na.rm = TRUE),
    clus36= sum(`36`, na.rm = TRUE),
    clus37= sum(`37`, na.rm = TRUE),
    clus38= sum(`38`, na.rm = TRUE),
    clus39= sum(`39`, na.rm = TRUE),
    clus40= sum(`40`, na.rm = TRUE),
    clus41= sum(`41`, na.rm = TRUE),
    clus42= sum(`42`, na.rm = TRUE),
    clus43= sum(`43`, na.rm = TRUE),
    clus44= sum(`44`, na.rm = TRUE),
    clus45= sum(`45`, na.rm = TRUE),
    clus46= sum(`46`, na.rm = TRUE),
    clus47= sum(`47`, na.rm = TRUE),
    clus48= sum(`48`, na.rm = TRUE),
    clus49= sum(`49`, na.rm = TRUE),
    clus50= sum(`50`, na.rm = TRUE)
  )


saveRDS(AC.d, "./data/work/dcluster_FR.RDS")
saveRDS(AC.n, "./data/work/ncluster_FR.RDS")

rm(list = ls())


##############################################################
##### UK
###### manage title encodings
### load encodings
te = read.csv2("./data/work/title_encodings_UK.csv",stringsAsFactors = FALSE)

### delete duplicates if any
te = te[which(!duplicated(te)),]

### convert vector dimensions columns to numeric
for (i in 6:ncol(te)){te[,i]=as.numeric(te[,i])}

### load url hits
load("./data/work/url_UK.RData")

### check that all encoded titles come from UK url hits by respondents
summary(te$url%in%url_UK$url)

### merge url hits with encodings
euUK = merge(url_UK[,c("url","panelist_id","used_at","domain")],te,by="url",all.x=TRUE)

### keep only one url hit by respondent
### to prevent overweighting articles manically read many many times by the same person
names(euUK)[which(names(euUK)=="panelist_id")]="pseudonym"
euUK = euUK[which(!duplicated(euUK[,c("pseudonym","url")])),]

### check results
print(paste("Successfully encoded urls:",floor(length(which(!is.na(euUK$X767)))/nrow(euUK)*100),"%"))

### filter out respondents for whom we don't have much info
### filter out articles whose titles have not been encoded
euUK$count = 1
individual.counts = aggregate(euUK$count,by=list(euUK$pseudonym),FUN=sum)
summary(individual.counts)
euUK = euUK[which(euUK$pseudonym %in% individual.counts$Group.1[which(individual.counts$x>10)]),]
euUK = na.omit(euUK)

####################
### PCA on BERT encoding
### interpreting dimensions and "who reads what"
for.PCA = euUK[,which(str_detect(names(euUK),"^X"))]
res = PCA(for.PCA)
dim = as.data.frame(res$ind$coord)
dim$index = 1:nrow(dim)
dim$pseudonym = euUK$pseudonym

### interpreting PCA dimensions
source("./euro-tracking/src/01_prep/0_2_6_1_helper_fun_topic_extraction.R")

### dimension 1: football vs scary news (abuse & crime)
interpret(dim,1,euUK)
### dimension 2: TV series vs Brexit
interpret(dim,2,euUK)
### dimension 3: Europen Elections vs (sex) crime
interpret(dim,3,euUK)
### dimension 4: Local news vs celebrities
interpret(dim,4,euUK)
### dimension 5: sports (non football, mixed with some other international stuff) vs deals (money related, business)
interpret(dim,5,euUK)

load("./data/orig/survey_data.RData")
mix = merge(dim,survey_data,by="pseudonym",all.x=TRUE)
mix$v_31 = factor(mix$v_31)
levels(mix$v_31) = c(NA,NA,NA,"remain","leave")
names(mix)[which(names(mix)=="v_31")]="referendum_vote"

### variable "change" says whether they changed their mind in the last European elections (pre-election vs post-election survey)
### variable "referendum_vote" says what they said they voted (if they voted) to the 2016 EU membership referendum

### referendum x topics: leave people are more into scary news, more into brexit news, and more into sex crime as well
aggregate(mix[,paste0("Dim.",c(1:5))],by=list(mix$referendum_vote),FUN=mean)

summary(lm(data = mix, family = gaussian, as.numeric(referendum_vote) ~ Dim.1 + Dim.2 + Dim.3 + Dim.4 + Dim.5 ))
summary(lm(data = mix, family = gaussian, as.numeric(as.factor(change)) ~ Dim.1 + Dim.2 + Dim.3 + Dim.4 + Dim.5 ))

################# determination du nombre optimal de cluster
encodings = euUK[,which(str_detect(names(euUK),"^X"))]
wss <- 2:50
for (i in 2:50) wss[i-1] <- sum(kmeans(encodings,
                                       centers=i)$withinss)
plot(2:50, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
################# creation des clusters dim 1 et 2 aprÃ¨s normalisation

nbclust <- 50
kmeans.comment <- kmeans(encodings,nbclust)
cluster.kmeans <- as.factor(kmeans.comment$cluster)
table(cluster.kmeans)
centers = kmeans.comment$centers
nrow(centers)

ac = data.frame(clusters = cluster.kmeans,pseudonym = euUK$pseudonym)
levels(ac$clusters) =c("general news","sports","Brexit","TV series","money","celebs","crime","football","entertainment")
ac$count = 1
ac$used_at <- euUK$used_at
ac$url <- euUK$url

url_UK.small <- url_UK %>% 
  select(panelist_id, url, used_at, active_seconds) %>% 
  rename(pseudonym = panelist_id)

AC <- merge(ac, url_UK.small, by=c("pseudonym", "url", "used_at"), all.x = TRUE)

AC.2 <- AC %>% 
  group_by(pseudonym, clusters) %>% 
  summarise(
    n_per_clust = n(),
    d_per_clust = sum(active_seconds, na.rm = TRUE))

AC.n <- spread(AC.2, clusters, n_per_clust)
AC.d <- spread(AC.2, clusters, d_per_clust)

AC.n <- AC.n %>%
  group_by(pseudonym) %>% 
  summarise(
    clus1 = sum(`1`, na.rm = TRUE),
    clus2 = sum(`2`, na.rm = TRUE),
    clus3 = sum(`3`, na.rm = TRUE),
    clus4 = sum(`4`, na.rm = TRUE),
    clus5 = sum(`5`, na.rm = TRUE),
    clus6 = sum(`6`, na.rm = TRUE),
    clus7 = sum(`7`, na.rm = TRUE),
    clus8 = sum(`8`, na.rm = TRUE),
    clus9 = sum(`9`, na.rm = TRUE),
    clus10 = sum(`10`, na.rm = TRUE),
    clus11 = sum(`11`, na.rm = TRUE),
    clus12 = sum(`12`, na.rm = TRUE),
    clus13 = sum(`13`, na.rm = TRUE),
    clus14 = sum(`14`, na.rm = TRUE),
    clus15 = sum(`15`, na.rm = TRUE),
    clus16 = sum(`16`, na.rm = TRUE),
    clus17 = sum(`17`, na.rm = TRUE),
    clus18 = sum(`18`, na.rm = TRUE),
    clus19 = sum(`19`, na.rm = TRUE),
    clus20 = sum(`20`, na.rm = TRUE),
    clus21 = sum(`21`, na.rm = TRUE),
    clus22 = sum(`22`, na.rm = TRUE),
    clus23 = sum(`23`, na.rm = TRUE),
    clus24 = sum(`24`, na.rm = TRUE),
    clus25 = sum(`25`, na.rm = TRUE),
    clus26 = sum(`26`, na.rm = TRUE),
    clus27 = sum(`27`, na.rm = TRUE),
    clus28 = sum(`28`, na.rm = TRUE),
    clus29 = sum(`29`, na.rm = TRUE),
    clus30 = sum(`30`, na.rm = TRUE),
    clus31 = sum(`31`, na.rm = TRUE),
    clus32 = sum(`32`, na.rm = TRUE),
    clus33= sum(`33`, na.rm = TRUE),
    clus34= sum(`34`, na.rm = TRUE),
    clus35= sum(`35`, na.rm = TRUE),
    clus36= sum(`36`, na.rm = TRUE),
    clus37= sum(`37`, na.rm = TRUE),
    clus38= sum(`38`, na.rm = TRUE),
    clus39= sum(`39`, na.rm = TRUE),
    clus40= sum(`40`, na.rm = TRUE),
    clus41= sum(`41`, na.rm = TRUE),
    clus42= sum(`42`, na.rm = TRUE),
    clus43= sum(`43`, na.rm = TRUE),
    clus44= sum(`44`, na.rm = TRUE),
    clus45= sum(`45`, na.rm = TRUE),
    clus46= sum(`46`, na.rm = TRUE),
    clus47= sum(`47`, na.rm = TRUE),
    clus48= sum(`48`, na.rm = TRUE),
    clus49= sum(`49`, na.rm = TRUE),
    clus50= sum(`50`, na.rm = TRUE)
  )

AC.d <- AC.d %>%
  group_by(pseudonym) %>% 
  summarise(
    clus1 = sum(`1`, na.rm = TRUE),
    clus2 = sum(`2`, na.rm = TRUE),
    clus3 = sum(`3`, na.rm = TRUE),
    clus4 = sum(`4`, na.rm = TRUE),
    clus5 = sum(`5`, na.rm = TRUE),
    clus6 = sum(`6`, na.rm = TRUE),
    clus7 = sum(`7`, na.rm = TRUE),
    clus8 = sum(`8`, na.rm = TRUE),
    clus9 = sum(`9`, na.rm = TRUE),
    clus10 = sum(`10`, na.rm = TRUE),
    clus11 = sum(`11`, na.rm = TRUE),
    clus12 = sum(`12`, na.rm = TRUE),
    clus13 = sum(`13`, na.rm = TRUE),
    clus14 = sum(`14`, na.rm = TRUE),
    clus15 = sum(`15`, na.rm = TRUE),
    clus16 = sum(`16`, na.rm = TRUE),
    clus17 = sum(`17`, na.rm = TRUE),
    clus18 = sum(`18`, na.rm = TRUE),
    clus19 = sum(`19`, na.rm = TRUE),
    clus20 = sum(`20`, na.rm = TRUE),
    clus21 = sum(`21`, na.rm = TRUE),
    clus22 = sum(`22`, na.rm = TRUE),
    clus23 = sum(`23`, na.rm = TRUE),
    clus24 = sum(`24`, na.rm = TRUE),
    clus25 = sum(`25`, na.rm = TRUE),
    clus26 = sum(`26`, na.rm = TRUE),
    clus27 = sum(`27`, na.rm = TRUE),
    clus28 = sum(`28`, na.rm = TRUE),
    clus29 = sum(`29`, na.rm = TRUE),
    clus30 = sum(`30`, na.rm = TRUE),
    clus31 = sum(`31`, na.rm = TRUE),
    clus32 = sum(`32`, na.rm = TRUE),
    clus33= sum(`33`, na.rm = TRUE),
    clus34= sum(`34`, na.rm = TRUE),
    clus35= sum(`35`, na.rm = TRUE),
    clus36= sum(`36`, na.rm = TRUE),
    clus37= sum(`37`, na.rm = TRUE),
    clus38= sum(`38`, na.rm = TRUE),
    clus39= sum(`39`, na.rm = TRUE),
    clus40= sum(`40`, na.rm = TRUE),
    clus41= sum(`41`, na.rm = TRUE),
    clus42= sum(`42`, na.rm = TRUE),
    clus43= sum(`43`, na.rm = TRUE),
    clus44= sum(`44`, na.rm = TRUE),
    clus45= sum(`45`, na.rm = TRUE),
    clus46= sum(`46`, na.rm = TRUE),
    clus47= sum(`47`, na.rm = TRUE),
    clus48= sum(`48`, na.rm = TRUE),
    clus49= sum(`49`, na.rm = TRUE),
    clus50= sum(`50`, na.rm = TRUE)
  )

saveRDS(AC.d, "./data/work/dcluster_UK.RDS")
saveRDS(AC.n, "./data/work/ncluster_UK.RDS")

