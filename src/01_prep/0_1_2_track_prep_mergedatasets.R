library(tidyverse)
setwd("/home/wrszemsrgyercqh/an_work/EU/")

URL1 <- readRDS("./data/orig/URL1.rds")
load("./data/orig/visits.RData")
sdata <- readRDS("./data/work/dat_surv.rds")

apps <- visits %>% 
  filter(d_kind == "app") %>% 
  rename(panelist_id = pseudonym,
         active_seconds = duration) %>% 
  select(-c(d_kind, pageviews, visit_id))
mobile.url <- visits %>% 
  filter(d_kind == "mobile") %>% 
  rename(panelist_id = pseudonym,
         active_seconds = duration) %>% 
  select(-c(d_kind, pageviews, visit_id))
pc.url <- URL1 %>% 
  select(-c(web_visits_id, id))

rm(visits, URL1)


sdata <- sdata %>% 
  select(panelist_id, country)

apps <- merge(apps, sdata,
              by="panelist_id",
              all = TRUE)
mobile.url <- merge(mobile.url, sdata,
              by="panelist_id",
              all = TRUE)
pc.url <- merge(pc.url, sdata,
              by="panelist_id",
              all = TRUE)

saveRDS(apps, "./data/work/apps.RDS")
saveRDS(mobile.url, "./data/work/m_url.RDS")
saveRDS(pc.url, "./data/work/pc_url.RDS")
