library(tidyverse)
setwd("/home/wrszemsrgyercqh/an_work/EU")

# Predictors: Domains

dur.dom.de <- readRDS("./data/work/dur_dom_de.RDS")
fre.dom.de <- readRDS("./data/work/fre_dom_de.RDS")
dur.dom.de[is.na(dur.dom.de)] <- 0
fre.dom.de[is.na(fre.dom.de)] <- 0


dur.dom.fr <- readRDS("./data/work/dur_dom_fr.RDS")
fre.dom.fr <- readRDS("./data/work/fre_dom_fr.RDS")
dur.dom.fr[is.na(dur.dom.fr)] <- 0
fre.dom.fr[is.na(fre.dom.fr)] <- 0


dur.dom.uk <- readRDS("./data/work/dur_dom_uk.RDS")
fre.dom.uk <- readRDS("./data/work/fre_dom_uk.RDS")
dur.dom.uk[is.na(dur.dom.uk)] <- 0
fre.dom.uk[is.na(fre.dom.uk)] <- 0

dur.dom.all <- readRDS("./data/work/dur_dom_all.RDS")
fre.dom.all <- readRDS("./data/work/fre_dom_all.RDS")
dur.dom.all[is.na(dur.dom.all)] <- 0
fre.dom.all[is.na(fre.dom.all)] <- 0

# Predictors: Apps

dur.app.de <- readRDS("./data/work/dur_app_de.RDS")
fre.app.de <- readRDS("./data/work/fre_app_de.RDS")
dur.app.de[is.na(dur.app.de)] <- 0
fre.app.de[is.na(fre.app.de)] <- 0

dur.app.fr <- readRDS("./data/work/dur_app_fr.RDS")
fre.app.fr <- readRDS("./data/work/fre_app_fr.RDS")
dur.app.fr[is.na(dur.app.fr)] <- 0
fre.app.fr[is.na(fre.app.fr)] <- 0

dur.app.uk <- readRDS("./data/work/dur_app_uk.RDS")
fre.app.uk <- readRDS("./data/work/fre_app_uk.RDS")
dur.app.uk[is.na(dur.app.uk)] <- 0
fre.app.uk[is.na(fre.app.uk)] <- 0

dur.app.all <- readRDS("./data/work/dur_app_all.RDS")
fre.app.all <- readRDS("./data/work/fre_app_all.RDS")
dur.app.all[is.na(dur.app.all)] <- 0
fre.app.all[is.na(fre.app.all)] <- 0

# Predictors: Categories

cat.app.de <- readRDS("./data/work/dur_per_cat_app_de.RDS")
cat.dom.de <- readRDS("./data/work/dur_per_cat_dom_de.RDS")

cat.app.de[is.na(cat.app.de)] <- 0
cat.dom.de[is.na(cat.dom.de)] <- 0


cat.app.fr <- readRDS("./data/work/dur_per_cat_app_fr.RDS")
cat.dom.fr <- readRDS("./data/work/dur_per_cat_dom_fr.RDS")

cat.app.fr[is.na(cat.app.fr)] <- 0
cat.dom.fr[is.na(cat.dom.fr)] <- 0


cat.app.uk <- readRDS("./data/work/dur_per_cat_app_uk.RDS")
cat.dom.uk <- readRDS("./data/work/dur_per_cat_dom_uk.RDS")

cat.app.uk[is.na(cat.app.uk)] <- 0
cat.dom.uk[is.na(cat.dom.uk)] <- 0

cat.app.all <- readRDS("./data/work/dur_per_cat_app_all.RDS")
cat.dom.all <- readRDS("./data/work/dur_per_cat_dom_all.RDS")

cat.app.all[is.na(cat.app.all)] <- 0
cat.dom.all[is.na(cat.dom.all)] <- 0

# Predictors: bert clusters
d.clust.de <- readRDS("./data/work/dcluster_DE.RDS")
d.clust.fr <- readRDS("./data/work/dcluster_FR.RDS")
d.clust.uk <- readRDS("./data/work/dcluster_UK.RDS")
# d.clust.all <- readRDS("./data/work/dcluster_ALL.RDS")


n.clust.de <- readRDS("./data/work/ncluster_DE.RDS")
n.clust.fr <- readRDS("./data/work/ncluster_FR.RDS")
n.clust.uk <- readRDS("./data/work/ncluster_UK.RDS")
# n.clust.all <- readRDS("./data/work/ncluster_ALL.RDS")

names(d.clust.de)[2:length(d.clust.de)] <- paste0("d.", names(d.clust.de[2:length(d.clust.de)]))
names(d.clust.fr)[2:length(d.clust.fr)] <- paste0("d.", names(d.clust.fr[2:length(d.clust.fr)]))
names(d.clust.uk)[2:length(d.clust.uk)] <- paste0("d.", names(d.clust.uk[2:length(d.clust.uk)]))
# names(d.clust.all)[2:length(d.clust.all)] <- paste0("d.", names(d.clust.all[2:length(d.clust.all)]))


names(n.clust.de)[2:length(n.clust.de)] <- paste0("n.", names(n.clust.de[2:length(n.clust.de)]))
names(n.clust.fr)[2:length(n.clust.fr)] <- paste0("n.", names(n.clust.fr[2:length(n.clust.fr)]))
names(n.clust.uk)[2:length(n.clust.uk)] <- paste0("n.", names(n.clust.uk[2:length(n.clust.uk)]))
# names(n.clust.all)[2:length(n.clust.all)] <- paste0("n.", names(n.clust.all[2:length(n.clust.all)]))



clust.de <- merge(d.clust.de,n.clust.de)
names(clust.de)[names(clust.de)=="pseudonym"] <- "panelist_id"
clust.fr <- merge(d.clust.fr,n.clust.fr)
names(clust.fr)[names(clust.fr)=="pseudonym"] <- "panelist_id"
clust.uk <- merge(d.clust.uk,n.clust.uk)
names(clust.uk)[names(clust.uk)=="pseudonym"] <- "panelist_id"
# clust.all <- merge(d.clust.all,n.clust.all)
# names(clust.all)[names(clust.all)=="pseudonym"] <- "panelist_id"



### DE
# dom predictors, dom users only
predictors.dom.de <- merge(dur.dom.de, fre.dom.de, by="panelist_id", all.x=TRUE)
# categories predictors, dom users only
predictors.cat.dom.de <- merge(cat.dom.de, predictors.dom.de[which(names(predictors.dom.de)=="panelist_id")], by="panelist_id", all.y=TRUE)

# app predictors, app users only
predictors.app.de <- merge(dur.app.de, fre.app.de, by="panelist_id", all.y=TRUE)
# categories predictors, app users only
predictors.cat.app.de <- merge(cat.app.de, predictors.app.de[which(names(predictors.app.de)=="panelist_id")], by="panelist_id", all.y=TRUE)

# bert predictors, bert users only
predictors.bert.de <- clust.de

### FR
# dom predictors, dom users only
predictors.dom.fr <- merge(dur.dom.fr, fre.dom.fr, by="panelist_id", all.x=TRUE)
# categories predictors, dom users only
predictors.cat.dom.fr <- merge(cat.dom.fr, predictors.dom.fr[which(names(predictors.dom.fr)=="panelist_id")], by="panelist_id", all.y=TRUE)

# app predictors, app users only
predictors.app.fr <- merge(dur.app.fr, fre.app.fr, by="panelist_id", all.y=TRUE)
# categories predictors, app users only
predictors.cat.app.fr <- merge(cat.app.fr, predictors.app.fr[which(names(predictors.app.fr)=="panelist_id")], by="panelist_id", all.y=TRUE)

# bert predictors, bert users only
predictors.bert.fr <- clust.fr

### UK
# dom predictors, dom users only
predictors.dom.uk <- merge(dur.dom.uk, fre.dom.uk, by="panelist_id", all.x=TRUE)
# categories predictors, dom users only
predictors.cat.dom.uk <- merge(cat.dom.uk, predictors.dom.uk[which(names(predictors.dom.uk)=="panelist_id")], by="panelist_id", all.y=TRUE)

# app predictors, app users only
predictors.app.uk <- merge(dur.app.uk, fre.app.uk, by="panelist_id", all.y=TRUE)
# categories predictors, app users only
predictors.cat.app.uk <- merge(cat.app.uk, predictors.app.uk[which(names(predictors.app.uk)=="panelist_id")], by="panelist_id", all.y=TRUE)

# bert predictors, bert users only
predictors.bert.uk <- clust.uk

### ALL
# dom predictors, dom users only
predictors.dom.all <- merge(dur.dom.all, fre.dom.all, by="panelist_id", all.x=TRUE)
# categories predictors, dom users only
predictors.cat.dom.all <- merge(cat.dom.all, predictors.dom.all[which(names(predictors.dom.all)=="panelist_id")], by="panelist_id", all.y=TRUE)

# app predictors, app users only
predictors.app.all <- merge(dur.app.all, fre.app.all, by="panelist_id", all.y=TRUE)
# categories predictors, app users only
predictors.cat.app.all <- merge(cat.app.all, predictors.app.all[which(names(predictors.app.all)=="panelist_id")], by="panelist_id", all.y=TRUE)

# bert predictors, bert users only --- dont have this yet
# predictors.bert.all <- clust.all



# convert to numeric in case there are factors
# ALL
indx <- sapply(predictors.app.all, is.factor)
predictors.app.all[indx] <- lapply(predictors.app.all[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.dom.all, is.factor)
predictors.dom.all[indx] <- lapply(predictors.dom.all[indx], function(x) as.numeric(as.character(x)))
rm(indx)
# indx <- sapply(predictors.bert.all, is.factor)
# predictors.bert.all[indx] <- lapply(predictors.bert.all[indx], function(x) as.numeric(as.character(x)))
# rm(indx)
predictors.cat.app.all$country <- NULL
indx <- sapply(predictors.cat.app.all, is.factor)
predictors.cat.app.all[indx] <- lapply(predictors.cat.app.all[indx], function(x) as.numeric(as.character(x)))
rm(indx)

predictors.cat.dom.all$country <- NULL
  indx <- sapply(predictors.cat.app.all, is.factor)
predictors.cat.app.all[indx] <- lapply(predictors.cat.app.all[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.cat.dom.all, is.factor)
predictors.cat.dom.all[indx] <- lapply(predictors.cat.dom.all[indx], function(x) as.numeric(as.character(x)))
rm(indx)


# reattach country var
ID <- cat.dom.all %>% 
  select(c(panelist_id, country))
ID <- ID[cat.dom.all$panelist_id %in% predictors.cat.dom.all$panelist_id,]
predictors.cat.dom.all <- merge(predictors.cat.dom.all, ID)

ID <- cat.app.all %>% 
  select(c(panelist_id, country))
ID <- ID[cat.app.all$panelist_id %in% predictors.cat.app.all$panelist_id,]
predictors.cat.app.all <- merge(predictors.cat.app.all, ID)



# de
indx <- sapply(predictors.app.de, is.factor)
predictors.app.de[indx] <- lapply(predictors.app.de[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.dom.de, is.factor)
predictors.dom.de[indx] <- lapply(predictors.dom.de[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.bert.de, is.factor)
predictors.bert.de[indx] <- lapply(predictors.bert.de[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.cat.app.de, is.factor)
predictors.cat.app.de[indx] <- lapply(predictors.cat.app.de[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.cat.dom.de, is.factor)
predictors.cat.dom.de[indx] <- lapply(predictors.cat.dom.de[indx], function(x) as.numeric(as.character(x)))
rm(indx)

# fr
indx <- sapply(predictors.app.fr, is.factor)
predictors.app.fr[indx] <- lapply(predictors.app.fr[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.dom.fr, is.factor)
predictors.dom.fr[indx] <- lapply(predictors.dom.fr[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.bert.fr, is.factor)
predictors.bert.fr[indx] <- lapply(predictors.bert.fr[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.cat.app.fr, is.factor)
predictors.cat.app.fr[indx] <- lapply(predictors.cat.app.fr[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.cat.dom.fr, is.factor)
predictors.cat.dom.fr[indx] <- lapply(predictors.cat.dom.fr[indx], function(x) as.numeric(as.character(x)))
rm(indx)

# uk
indx <- sapply(predictors.app.uk, is.factor)
predictors.app.uk[indx] <- lapply(predictors.app.uk[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.dom.uk, is.factor)
predictors.dom.uk[indx] <- lapply(predictors.dom.uk[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.bert.uk, is.factor)
predictors.bert.uk[indx] <- lapply(predictors.bert.uk[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.cat.app.uk, is.factor)
predictors.cat.app.uk[indx] <- lapply(predictors.cat.app.uk[indx], function(x) as.numeric(as.character(x)))
rm(indx)
indx <- sapply(predictors.cat.dom.uk, is.factor)
predictors.cat.dom.uk[indx] <- lapply(predictors.cat.dom.uk[indx], function(x) as.numeric(as.character(x)))
rm(indx)


# change any NAs to 0 ... NAs result from merge of much smaller app dataset with larger domains dataset
predictors.app.uk[is.na(predictors.app.uk)] <- 0
predictors.app.fr[is.na(predictors.app.fr)] <- 0
predictors.app.de[is.na(predictors.app.de)] <- 0
predictors.app.all[is.na(predictors.app.all)] <- 0



predictors.cat.app.uk[is.na(predictors.cat.app.uk)] <- 0
predictors.cat.app.fr[is.na(predictors.cat.app.fr)] <- 0
predictors.cat.app.de[is.na(predictors.cat.app.de)] <- 0
predictors.cat.app.all[is.na(predictors.cat.app.all)] <- 0



predictors.cat.dom.uk[is.na(predictors.cat.dom.uk)] <- 0
predictors.cat.dom.fr[is.na(predictors.cat.dom.fr)] <- 0
predictors.cat.dom.de[is.na(predictors.cat.dom.de)] <- 0
predictors.cat.dom.all[is.na(predictors.cat.dom.all)] <- 0



predictors.dom.uk[is.na(predictors.dom.uk)] <- 0
predictors.dom.fr[is.na(predictors.dom.fr)] <- 0
predictors.dom.de[is.na(predictors.dom.de)] <- 0
predictors.dom.all[is.na(predictors.dom.all)] <- 0



predictors.bert.uk[is.na(predictors.bert.uk)] <- 0
predictors.bert.fr[is.na(predictors.bert.fr)] <- 0
predictors.bert.de[is.na(predictors.bert.de)] <- 0
# predictors.bert.all[is.na(predictors.bert.all)] <- 0


saveRDS(predictors.bert.uk, file="./data/work/pred_track_bert_uk.RDS")
saveRDS(predictors.bert.fr, file="./data/work/pred_track_bert_fr.RDS")
saveRDS(predictors.bert.de, file="./data/work/pred_track_bert_de.RDS")
# saveRDS(predictors.bert.all, file="./data/work/pred_track_bert_all.RDS")



saveRDS(predictors.dom.uk, file="./data/work/pred_track_dom_uk.RDS")
saveRDS(predictors.dom.fr, file="./data/work/pred_track_dom_fr.RDS")
saveRDS(predictors.dom.de, file="./data/work/pred_track_dom_de.RDS")
saveRDS(predictors.dom.all, file="./data/work/pred_track_dom_all.RDS")


saveRDS(predictors.cat.dom.uk, file="./data/work/pred_track_cat_dom_uk.RDS")
saveRDS(predictors.cat.dom.fr, file="./data/work/pred_track_cat_dom_fr.RDS")
saveRDS(predictors.cat.dom.de, file="./data/work/pred_track_cat_dom_de.RDS")
saveRDS(predictors.cat.dom.all, file="./data/work/pred_track_cat_dom_all.RDS")


saveRDS(predictors.app.uk, file="./data/work/pred_track_app_uk.RDS")
saveRDS(predictors.app.fr, file="./data/work/pred_track_app_fr.RDS")
saveRDS(predictors.app.de, file="./data/work/pred_track_app_de.RDS")
saveRDS(predictors.app.all, file="./data/work/pred_track_app_all.RDS")


saveRDS(predictors.cat.app.uk, file="./data/work/pred_track_cat_app_uk.RDS")
saveRDS(predictors.cat.app.fr, file="./data/work/pred_track_cat_app_fr.RDS")
saveRDS(predictors.cat.app.de, file="./data/work/pred_track_cat_app_de.RDS")
saveRDS(predictors.cat.app.all, file="./data/work/pred_track_cat_app_all.RDS")

