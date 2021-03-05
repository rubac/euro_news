library(tidyverse)
setwd("/home/wrszemsrgyercqh/an_work/EU/")

pc.url <- readRDS("./data/work/pc_url.RDS")
m.url <- readRDS("./data/work/m_url.RDS")

pc.url <- pc.url %>% 
  select(-c(url))

URL1 <- rbind(pc.url, m.url)
rm(pc.url, m.url)


# Need to make sure we only have logs before election
URL1 <- URL1 %>% 
  filter(used_at<="2019-05-26 00:00:01")


# split "category" into four (max number of categories in "category") separate vars
cat1  <-  separate(URL1, category, c("cat1", NA, NA, NA), convert = TRUE)
cat1 <- cat1 %>% select(cat1)
cat2  <-  separate(URL1, category, c(NA, "cat2", NA, NA), convert = TRUE)
cat2 <- cat2 %>% select(cat2)
cat3  <-  separate(URL1, category, c(NA, NA, "cat3", NA), convert = TRUE)
cat3 <- cat3 %>% select(cat3)
cat4  <-  separate(URL1, category, c(NA, NA, NA, "cat4"), convert = TRUE)
cat4 <- cat4 %>% select(cat4)

categories <- cbind(cat1$cat1, cat2$cat2, cat3$cat3, cat4$cat4)
rm(cat1, cat2, cat3, cat4)
categories <- as.data.frame(categories)
URL1 <- cbind(URL1, categories)
rm(categories)
URL1 <- URL1  %>%
  select(-category) %>% 
  rename(cat1 = V1,
         cat2 = V2,
         cat3 = V3,
         cat4 = V4
  )

# Code NA as factor level
X <- apply(data.frame(URL1$cat1, URL1$cat2, URL1$cat3, URL1$cat4), 2,  fct_explicit_na)
X <- as.data.frame(X)
URL1$cat1 <- X$URL1.cat1
URL1$cat2 <- X$URL1.cat2
URL1$cat3 <- X$URL1.cat3
URL1$cat4 <- X$URL1.cat4
rm(X)


levels(URL1$cat1) <- c("missing", "abortion", "adult", "advertising", "alcoholandtobacco", "blacklist",
                       "blogsandpersonal", "business", "chatandmessaging","contentserver", "dating", "dating",
                       "deceptive", "drugs", "economyandfinance", "education", "entertainment", "foodandrecipes",
                       "foodandrecipes", "foodandrecipes", "gambling", "games", "health", "illegalcontent",
                       "informationtech", "jobrelated", "malicious", "mediasharing", "messageboardsandforums",
                       "newsandmedia", "newsandmedia", "parked", "personals", "proxyandfilteravoidance",
                       "realestate", "religion", "searchenginesandportals", "shopping", "socialnetworking",
                       "sports", "sports", "streamingmedia", "translation", "translation", "travel", "uncategorized",
                       "vehicles", "weapons")

levels(URL1$cat2) <- c("missing", "missing", "adult", "advertising", "alcoholandtobacco", "blogsandpersonal",
                       "business", "business", "chatandmessaging", "contentserver", "deceptive", "drugs", 
                       "economyandfinance", "education", "entertainment", "foodandrecipes", "gambling", 
                       "games", "health", "humor", "illegalcontent", "informationtech", "jobrelated",
                       "mediasharing", "messageboardsandforums", "newsandmedia", "parked", "personals",
                       "proxyandfilteravoidance", "realestate", "religion", "searchenginesandportals",
                       "shopping", "socialnetworking", "sports", "streamingmedia", "translation", "travel",
                       "vehicles", "virtualreality", "weapons")


levels(URL1$cat3) <- c("missing", "adult", "advertising", "alcoholandtobacco", "blogsandpersonal", "business",
                       "chatandmessaging", "drugs", "economyandfinance", "education", "entertainment", "foodandrecipes",
                       "gambling", "games", "health", "illegalcontent", "informationtech", "jobrelated", "mediasharing",
                       "messageboardsandforums", "newsandmedia", "parked", "personals", "proxyandfilteravoidance",
                       "realestate", "religion", "searchenginesandportals", "shopping", "socialnetworking", "sports",
                       "streamingmedia", "translation", "travel", "vehicles", "virtualreality", "weapons")


levels(URL1$cat4) <- c("missing", "alcoholandtobacco", "blogsandpersonal", "business", "education", "shopping", "vehicles")
##################################################################################################################
#### create variables from categories of URLs: calculate duration and frequency of use per category


duration1 <- URL1 %>%
  group_by(panelist_id, cat1) %>%
  summarise(
    d_per_cat1 = sum(active_seconds, na.rm = TRUE),
    n_per_cat1 = n()
  ) %>% 
  ungroup()

duration2 <- URL1 %>%
  group_by(panelist_id, cat2) %>%
  summarise(
    d_per_cat2 = sum(active_seconds, na.rm = TRUE),
    n_per_cat2 = n()
  ) %>% 
  ungroup()


duration3 <- URL1 %>%
  group_by(panelist_id, cat3) %>%
  summarise(
    d_per_cat3 = sum(active_seconds, na.rm = TRUE),
    n_per_cat3 = n()
  )  %>% 
  ungroup()


duration4 <- URL1 %>%
  group_by(panelist_id, cat4) %>%
  summarise(
    d_per_cat4 = sum(active_seconds, na.rm = TRUE),
    n_per_cat4 = n()
  )  %>% 
  ungroup()


# for duration 1
duration1.1 <- select(duration1, -n_per_cat1)
duration1.2 <- select(duration1, -d_per_cat1)


duration1.1 <- spread(duration1.1, cat1, d_per_cat1)
duration1.2 <- spread(duration1.2, cat1, n_per_cat1)

names1.1 <- names(duration1.1)
names1.2 <- names(duration1.2)

names1.1 <- paste0('d_' , names1.1, "1")
names1.2 <- paste0('n_' , names1.2, "1")

duration1.1 <- set_names(duration1.1, nm = names1.1)
duration1.2 <- set_names(duration1.2, nm = names1.2)
rm(names1.1,names1.2)


duration1.1 <- rename(duration1.1, panelist_id = d_panelist_id1)
duration1.2 <- rename(duration1.2, panelist_id = n_panelist_id1)

# for duration 2
duration2.1 <- select(duration2, -n_per_cat2)
duration2.2 <- select(duration2, -d_per_cat2)


duration2.1 <- spread(duration2.1, cat2, d_per_cat2)
duration2.2 <- spread(duration2.2, cat2, n_per_cat2)

names2.1 <- names(duration2.1)
names2.2 <- names(duration2.2)

names2.1 <- paste0('d_' , names2.1, "2")
names2.2 <- paste0('n_' , names2.2, "2")

duration2.1 <- set_names(duration2.1, nm = names2.1)
duration2.2 <- set_names(duration2.2, nm = names2.2)
rm(names2.1,names2.2)


duration2.1 <- rename(duration2.1, panelist_id = d_panelist_id2)
duration2.2 <- rename(duration2.2, panelist_id = n_panelist_id2)

# for duration 3
duration3.1 <- select(duration3, -n_per_cat3)
duration3.2 <- select(duration3, -d_per_cat3)


duration3.1 <- spread(duration3.1, cat3, d_per_cat3)
duration3.2 <- spread(duration3.2, cat3, n_per_cat3)

names3.1 <- names(duration3.1)
names3.2 <- names(duration3.2)

names3.1 <- paste0('d_' , names3.1, "3")
names3.2 <- paste0('n_' , names3.2, "3")

duration3.1 <- set_names(duration3.1, nm = names3.1)
duration3.2 <- set_names(duration3.2, nm = names3.2)
rm(names3.1,names3.2)


duration3.1 <- rename(duration3.1, panelist_id = d_panelist_id3)
duration3.2 <- rename(duration3.2, panelist_id = n_panelist_id3)

# for duration 4
duration4.1 <- select(duration4, -n_per_cat4)
duration4.2 <- select(duration4, -d_per_cat4)


duration4.1 <- spread(duration4.1, cat4, d_per_cat4)
duration4.2 <- spread(duration4.2, cat4, n_per_cat4)

names4.1 <- names(duration4.1)
names4.2 <- names(duration4.2)

names4.1 <- paste0('d_' , names4.1, "4")
names4.2 <- paste0('n_' , names4.2, "4")

duration4.1 <- set_names(duration4.1, nm = names4.1)
duration4.2 <- set_names(duration4.2, nm = names4.2)
rm(names4.1,names4.2)


duration4.1 <- rename(duration4.1, panelist_id = d_panelist_id4)
duration4.2 <- rename(duration4.2, panelist_id = n_panelist_id4)


duration1 <- merge(duration1.1, duration1.2,
                  by="panelist_id",
                  all = TRUE)
duration2 <- merge(duration2.1, duration2.2,
                  by="panelist_id",
                  all = TRUE)
duration3 <- merge(duration3.1, duration3.2,
                  by="panelist_id",
                  all = TRUE)
duration4 <- merge(duration4.1, duration4.2,
                  by="panelist_id",
                  all = TRUE)

rm(duration1.1,duration1.2,duration2.1,duration2.2,duration3.1,duration3.2,duration4.1,duration4.2)

duration <- merge(duration1, duration2,
                   by="panelist_id",
                   all = TRUE)
duration <- merge(duration, duration3,
                  by="panelist_id",
                  all = TRUE)
duration <- merge(duration, duration4,
                  by="panelist_id",
                  all = TRUE)

rm(duration1,duration2,duration3,duration4)

# total online time
total_dur <- URL1 %>%
  group_by(panelist_id) %>%
  summarise(
    d_total = sum(active_seconds, na.rm = TRUE),
    n_total = n()
  ) %>% 
  ungroup()

duration <- merge(total_dur, duration, by="panelist_id", all = TRUE)
rm(total_dur)

# calculate share of news_media/total online time
duration[is.na(duration)]  <- 0

duration$d.d_newsandmedia <- (duration$d_newsandmedia1 + duration$d_newsandmedia2 + duration$d_newsandmedia3)
duration$d.n_newsandmedia <- (duration$n_newsandmedia1 + duration$n_newsandmedia2 + duration$n_newsandmedia3)

duration$d.rd_newsandmedia <- duration$d.d_newsandmedia/duration$d_total
duration$d.rn_newsandmedia <- duration$d.n_newsandmedia/duration$n_total

summary(duration$d.rd_newsandmedia)

summary(duration$d.rn_newsandmedia)

X <- readRDS("./data/work/dat_surv.rds")
X <- X %>% 
  select(c(panelist_id, country))

duration <- merge(duration, X, by="panelist_id", all.x = TRUE)

duration <- duration %>% 
  rename(d.total.dcat = d_total,
         n.total.dcat = n_total)

duration.de <- duration %>% 
  filter(country == "Germany") %>% 
  select(-country)
duration.fr <- duration %>% 
  filter(country == "France") %>% 
  select(-country)
duration.uk <- duration %>% 
  filter(country == "UK") %>% 
  select(-country)


saveRDS(duration.uk, "./data/work/dur_per_cat_dom_uk.RDS")
saveRDS(duration.de, "./data/work/dur_per_cat_dom_de.RDS")
saveRDS(duration.fr, "./data/work/dur_per_cat_dom_fr.RDS")
saveRDS(duration, "./data/work/dur_per_cat_dom_all.RDS")

rm(list = ls())

# do the same with apps
app <- readRDS("./data/work/apps.RDS")

# remove NAs
app <- app[!(is.na(app$domain)),]

app <- app %>% 
  filter(used_at<="2019-05-26 00:00:01")

app$category <- as.factor(app$category)

levels(app$category) <- c("adult", "entertainment", "business", "messaging", "finance", "games",
                       "health", "sports", "jobsandedu","lifestyle", "newsmedia", "preinstalled",
                       "search", "shopping", "social", "tools", "travel", "weather")


duration <- app %>%
  group_by(panelist_id, category) %>%
  summarise(
    d_per_cat = sum(active_seconds, na.rm = TRUE),
    n_per_cat = n()
  ) %>% 
  ungroup()

duration.1 <- select(duration, -n_per_cat)
duration.2 <- select(duration, -d_per_cat)


duration.1 <- spread(duration.1, category, d_per_cat)
duration.2 <- spread(duration.2, category, n_per_cat)

names.1 <- names(duration.1)
names.2 <- names(duration.2)

names.1 <- paste0('d_' , names.1)
names.2 <- paste0('n_' , names.2)

duration.1 <- set_names(duration.1, nm = names.1)
duration.2 <- set_names(duration.2, nm = names.2)
rm(names.1,names.2)


duration.1 <- rename(duration.1, panelist_id = d_panelist_id)
duration.2 <- rename(duration.2, panelist_id = n_panelist_id)


duration <- merge(duration.1, duration.2,
                   by="panelist_id",
                   all = TRUE)

rm(duration.1,duration.2)




# total online time
total_dur <- app %>%
  group_by(panelist_id) %>%
  summarise(
    d_total = sum(active_seconds, na.rm = TRUE),
    n_total = n()
  ) %>% 
  ungroup()

duration <- merge(total_dur, duration, by="panelist_id", all = TRUE)
rm(total_dur)

# calculate share of news_media/total online time
duration[is.na(duration)]  <- 0

duration$a.rd_newsandmedia <- duration$d_newsmedia/duration$d_total
duration$a.rn_newsandmedia <- duration$n_newsmedia/duration$n_total

summary(duration$a.rd_newsandmedia)

summary(duration$a.rn_newsandmedia)

X <- readRDS("./data/work/dat_surv.rds")
X <- X %>% 
  select(c(panelist_id, country))

duration <- merge(duration, X, by="panelist_id", all.x = TRUE)
duration <- duration %>% 
  rename(d.total.acat = d_total,
         n.total.acat = n_total)

duration.de <- duration %>% 
  filter(country == "Germany") %>% 
  select(-country)
duration.fr <- duration %>% 
  filter(country == "France") %>% 
  select(-country)
duration.uk <- duration %>% 
  filter(country == "UK") %>% 
  select(-country)


saveRDS(duration.uk, "./data/work/dur_per_cat_app_uk.RDS")
saveRDS(duration.de, "./data/work/dur_per_cat_app_de.RDS")
saveRDS(duration.fr, "./data/work/dur_per_cat_app_fr.RDS")
saveRDS(duration, "./data/work/dur_per_cat_app_all.RDS")






