library(tidyverse)
setwd("/home/wrszemsrgyercqh/an_work/EU")

app <- readRDS("./data/work/apps.RDS")
app <- app[!(is.na(app$domain)),]

# Need to make sure we only have logs before survey data collection
app <- app %>% 
  filter(used_at<="2019-05-17 00:00:01")

# remove system apps
drop.app <- read.csv("./data/work/system_apps.csv")
drop.app <- drop.app %>% 
  rename(domain = n_packagenamestring) %>% 
  select(domain)
app <- app[!(app$domain %in% drop.app$domain), ]
rm(drop.app)


#### all countries

app.all <- app
### Limit to apps used by at least 3 different people
# calculate number of visits to each app by each respondent
app.all.keep <- app.all %>% 
  group_by(domain, panelist_id) %>%
  summarise(
    n_per_app = n()
  )

# replace n_per_dom with one
app.all.keep$n_per_app <- 1
app.all.keep <- app.all.keep[!is.na(app.all.keep$domain),]

length((unique(app.all.keep$domain)))
length((unique(app.all.keep$panelist_id)))

# calculate again how many. now, its a count of how many diff participants per app (because n_per_dom==1)
# delete if smaller/equal  3
app.all.keep <- app.all.keep %>% 
  group_by(domain) %>% 
  summarise(
    n_users = n()
  ) %>% 
  filter(n_users>=50) %>% 
  select(domain)
length(unique(app.all.keep$domain))

# df withoout apps visited by fewer than 3 diff people
app.all.reduced <- app.all[app.all$domain %in% app.all.keep$domain, ]
rm(app.all.keep)

# df to remove those used fewer than 20 times and less than 60 seconds
# basis is already reduced data set app.all.reduced
app.all.keep.2 <- app.all.reduced %>% 
  group_by(domain) %>%
  summarise(
    d_per_app = sum(active_seconds, na.rm = TRUE),
    n_per_app = n()
  ) %>% 
  filter(d_per_app >= 1200 & n_per_app >= 300) %>% 
  ungroup() %>% 
  select(domain) # keep only apps

length(unique(app.all.keep.2$domain))
# delete from reduced dataset
app.all.reduced <- app.all.reduced[app.all.reduced$domain %in% app.all.keep.2$domain, ]
rm(app.all.keep.2)
## finally,  calculate duration and frequency per app (only those that remain)
app.all.dur <- app.all.reduced %>% 
  group_by(panelist_id, domain) %>%
  summarise(
    d_per_app = sum(active_seconds, na.rm = TRUE),
    n_per_app = n()
  ) %>%
  ungroup() 
# number of apps
length(unique(app.all.dur$domain))
# number of participants
length(unique(app.all.dur$panelist_id))

weird.ones.all <- app.all[(!(app.all$panelist_id %in% app.all.dur$panelist_id)), ]



##################################################################################################################
#### for each country, create vars containing duration and frequency for most visited apps

## UK
app.uk <- app %>%
  filter(country=="UK")
#how many participatns?
length(unique(app.uk$panelist_id))
#how many different apps?
length(unique(app.uk$domain))

### Limit to apps used by at least 3 different people
# calculate number of visits to each app by each respondent
app.uk.keep <- app.uk %>% 
  group_by(domain, panelist_id) %>%
  summarise(
    n_per_app = n()
  )

# replace n_per_dom with one
app.uk.keep$n_per_app <- 1
app.uk.keep <- app.uk.keep[!is.na(app.uk.keep$domain),]

length((unique(app.uk.keep$domain)))
length((unique(app.uk.keep$panelist_id)))

# calculate again how many. now, its a count of how many diff participants per app (because n_per_dom==1)
# delete if smaller/equal  3
app.uk.keep <- app.uk.keep %>% 
  group_by(domain) %>% 
  summarise(
    n_users = n()
  ) %>% 
  filter(n_users>=50) %>% 
  select(domain)
length(unique(app.uk.keep$domain))

# df withoout apps visited by fewer than 3 diff people
app.uk.reduced <- app.uk[app.uk$domain %in% app.uk.keep$domain, ]
rm(app.uk.keep)

# df to remove those used fewer than 20 times and less than 60 seconds
# basis is already reduced data set app.uk.reduced
app.uk.keep.2 <- app.uk.reduced %>% 
  group_by(domain) %>%
  summarise(
    d_per_app = sum(active_seconds, na.rm = TRUE),
    n_per_app = n()
  ) %>% 
  filter(d_per_app >= 1200 & n_per_app >= 300) %>% 
  ungroup() %>% 
  select(domain) # keep only apps

length(unique(app.uk.keep.2$domain))
# delete from reduced dataset
app.uk.reduced <- app.uk.reduced[app.uk.reduced$domain %in% app.uk.keep.2$domain, ]
rm(app.uk.keep.2)
## finally,  calculate duration and frequency per app (only those that remain)
app.uk.dur <- app.uk.reduced %>% 
  group_by(panelist_id, domain) %>%
  summarise(
    d_per_app = sum(active_seconds, na.rm = TRUE),
    n_per_app = n()
  ) %>%
  ungroup() 
# number of apps
length(unique(app.uk.dur$domain))
# number of participants
length(unique(app.uk.dur$panelist_id))

weird.ones.uk <- app.uk[(!(app.uk$panelist_id %in% app.uk.dur$panelist_id)), ]


## France
app.fr <- app %>%
  filter(country=="France")
#how many participatns?
length(unique(app.fr$panelist_id))
#how many different apps?
length(unique(app.fr$domain))

### Limit to apps used by at least 3 different people
# calculate number of visits to each apps by each respondent
app.fr.keep <- app.fr %>% 
  group_by(domain, panelist_id) %>%
  summarise(
    n_per_app = n()
  )

# replace n_per_dom with one
app.fr.keep$n_per_dom <- 1
app.fr.keep <- app.fr.keep[!is.na(app.fr.keep$domain),]

length((unique(app.fr.keep$domain)))
length((unique(app.fr.keep$panelist_id)))

# calculate again how many. now, its a count of how many diff participants per apps (because n_per_dom==1)
# delete if smaller/equal  3
app.fr.keep <- app.fr.keep %>% 
  group_by(domain) %>% 
  summarise(
    n_users = n()
  ) %>% 
  filter(n_users>=50) %>% 
  select(domain)
length(unique(app.fr.keep$domain))

# df withoout apps visited by fewer than 3 diff people
app.fr.reduced <- app.fr[app.fr$domain %in% app.fr.keep$domain, ]
rm(app.fr.keep)

# df to remove those visited fewer than 20 times and less than 60 seconds
# basis is already reduced data set app.fr.reduced
app.fr.keep.2 <- app.fr.reduced %>% 
  group_by(domain) %>%
  summarise(
    d_per_app = sum(active_seconds, na.rm = TRUE),
    n_per_app = n()
  ) %>% 
  filter(d_per_app >= 1200 & n_per_app >= 300) %>% 
  ungroup() %>% 
  select(domain) # keep only apps

length(unique(app.fr.keep.2$domain))
# delete from reduced dataset
app.fr.reduced <- app.fr.reduced[app.fr.reduced$domain %in% app.fr.keep.2$domain, ]
rm(app.fr.keep.2)
## finally,  calculate duration and frequency per apps (only those that remain)
app.fr.dur <- app.fr.reduced %>% 
  group_by(panelist_id, domain) %>%
  summarise(
    d_per_app = sum(active_seconds, na.rm = TRUE),
    n_per_app = n()
  ) %>%
  ungroup() 
# number of apps
length(unique(app.fr.dur$domain))
# number of participants
length(unique(app.fr.dur$panelist_id))

weird.ones.fr <- app.fr[(!(app.fr$panelist_id %in% app.fr.dur$panelist_id)), ]


## Germany
app.de <- app %>%
  filter(country=="Germany")
#how many participatns?
length(unique(app.de$panelist_id))
#how many different app?
length(unique(app.de$domain))

### Limit to app used by at least 3 different people
# calculate number of visits to each app by each respondent
app.de.keep <- app.de %>% 
  group_by(domain, panelist_id) %>%
  summarise(
    n_per_app = n()
  )

# replace n_per_app with one
app.de.keep$n_per_dom <- 1
app.de.keep <- app.de.keep[!is.na(app.de.keep$domain),]

length((unique(app.de.keep$domain)))
length((unique(app.de.keep$panelist_id)))

# calculate again how many. now, its a count of how many diff participants per app (because n_per_dom==1)
# delete if smaller/equal  3
app.de.keep <- app.de.keep %>% 
  group_by(domain) %>% 
  summarise(
    n_users = n()
  ) %>% 
  filter(n_users>=50) %>% 
  select(domain)
length(unique(app.de.keep$domain))

# df withoout app visited by fewer than 3 diff people
app.de.reduced <- app.de[app.de$domain %in% app.de.keep$domain, ]
rm(app.de.keep)

# df to remove those visited fewer than 20 times and less than 60 seconds
# basis is already reduced data set app.de.reduced
app.de.keep.2 <- app.de.reduced %>% 
  group_by(domain) %>%
  summarise(
    d_per_app = sum(active_seconds, na.rm = TRUE),
    n_per_app = n()
  ) %>% 
  filter(d_per_app >= 1200 & n_per_app >= 300) %>% 
  ungroup() %>% 
  select(domain) # keep only app

length(unique(app.de.keep.2$domain))
# delete from reduced dataset
app.de.reduced <- app.de.reduced[app.de.reduced$domain %in% app.de.keep.2$domain, ]
rm(app.de.keep.2)
## finally,  calculate duration and frequency per app (only those that remain)
app.de.dur <- app.de.reduced %>% 
  group_by(panelist_id, domain) %>%
  summarise(
    d_per_app = sum(active_seconds, na.rm = TRUE),
    n_per_app = n()
  ) %>%
  ungroup() 
# number of domains
length(unique(app.de.dur$domain))
# number of participants
length(unique(app.de.dur$panelist_id))

weird.ones.de <- app.de[(!(app.de$panelist_id %in% app.de.dur$panelist_id)), ]



### #remove what is obsolete
rm(app.de.reduced,app.fr.reduced,app.uk.reduced)

overlap <- app.de.dur[app.de.dur$domain %in% app.fr.dur$domain,]
overlap <- overlap[overlap$domain %in% app.uk.dur$domain,]
overlap <- overlap %>% 
  distinct(domain) %>% 
  select(domain)

length(overlap$domain) # only 19 overlapping apps between countries

#######calculate time and frequency per app

# ALL
app.all.dur <- app.all.dur[app.all.dur$domain %in% overlap$domain,]
dur.all <- select(app.all.dur, -n_per_app)
fre.all <- select(app.all.dur, -d_per_app)


dur.all <- spread(dur.all, domain, d_per_app)
fre.all <- spread(fre.all, domain, n_per_app)

names.dur.all <- names(dur.all)
names.fre.all <- names(fre.all)

names.dur.all <- paste0('d_' , names.dur.all)
names.fre.all <- paste0('n_' , names.fre.all)

dur.all <- set_names(dur.all, nm = names.dur.all)
fre.all <- set_names(fre.all, nm = names.fre.all)
rm(names.dur.all,names.fre.all)


dur.all <- rename(dur.all, panelist_id = d_panelist_id)
fre.all <- rename(fre.all, panelist_id = n_panelist_id)


# DE
dur.de <- select(app.de.dur, -n_per_app)
fre.de <- select(app.de.dur, -d_per_app)


dur.de <- spread(dur.de, domain, d_per_app)
fre.de <- spread(fre.de, domain, n_per_app)

names.dur.de <- names(dur.de)
names.fre.de <- names(fre.de)

names.dur.de <- paste0('d_' , names.dur.de)
names.fre.de <- paste0('n_' , names.fre.de)

dur.de <- set_names(dur.de, nm = names.dur.de)
fre.de <- set_names(fre.de, nm = names.fre.de)
rm(names.dur.de,names.fre.de)


dur.de <- rename(dur.de, panelist_id = d_panelist_id)
fre.de <- rename(fre.de, panelist_id = n_panelist_id)

# FR
dur.fr <- select(app.fr.dur, -n_per_app)
fre.fr <- select(app.fr.dur, -d_per_app)


dur.fr <- spread(dur.fr, domain, d_per_app)
fre.fr <- spread(fre.fr, domain, n_per_app)

names.dur.fr <- names(dur.fr)
names.fre.fr <- names(fre.fr)

names.dur.fr <- paste0('d_' , names.dur.fr)
names.fre.fr <- paste0('n_' , names.fre.fr)

dur.fr <- set_names(dur.fr, nm = names.dur.fr)
fre.fr <- set_names(fre.fr, nm = names.fre.fr)
rm(names.dur.fr,names.fre.fr)


dur.fr <- rename(dur.fr, panelist_id = d_panelist_id)
fre.fr <- rename(fre.fr, panelist_id = n_panelist_id)

# UK
dur.uk <- select(app.uk.dur, -n_per_app)
fre.uk <- select(app.uk.dur, -d_per_app)


dur.uk <- spread(dur.uk, domain, d_per_app)
fre.uk <- spread(fre.uk, domain, n_per_app)

names.dur.uk <- names(dur.uk)
names.fre.uk <- names(fre.uk)

names.dur.uk <- paste0('d_' , names.dur.uk)
names.fre.uk <- paste0('n_' , names.fre.uk)

dur.uk <- set_names(dur.uk, nm = names.dur.uk)
fre.uk <- set_names(fre.uk, nm = names.fre.uk)
rm(names.dur.uk,names.fre.uk)


dur.uk <- rename(dur.uk, panelist_id = d_panelist_id)
fre.uk <- rename(fre.uk, panelist_id = n_panelist_id)


saveRDS(dur.de, "./data/work/dur_app_de.RDS")
saveRDS(fre.de, "./data/work/fre_app_de.RDS")

saveRDS(dur.fr, "./data/work/dur_app_fr.RDS")
saveRDS(fre.fr, "./data/work/fre_app_fr.RDS")

saveRDS(dur.uk, "./data/work/dur_app_uk.RDS")
saveRDS(fre.uk, "./data/work/fre_app_uk.RDS")

saveRDS(dur.all, "./data/work/dur_app_all.RDS")
saveRDS(fre.all, "./data/work/fre_app_all.RDS")





