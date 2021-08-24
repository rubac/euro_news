library(tidyverse)
setwd("/home/wrszemsrgyercqh/an_work/EU")

pc.url <- readRDS("./data/work/pc_url.RDS")
m.url <- readRDS("./data/work/m_url.RDS")

pc.url <- pc.url %>% 
  select(-c(url))

URL1 <- rbind(pc.url, m.url)
rm(pc.url, m.url)

# Need to make sure we only have logs before survey data collection
URL1 <- URL1 %>% 
  filter(used_at<="2019-05-17 00:00:01")

# remove ending of domains to create overlap between countries
URL1$domain <- gsub("\\..*","",URL1$domain)


URL1.allcountry <- URL1

##################################################################################################################
#### all countries, create vars containing duration and frequency for most visited domains

### Limit to sites used by at least 50 different people
# calculate number of visits to each domain by each respondent
URL1.allcountry.keep <- URL1.allcountry %>% 
  group_by(domain, panelist_id) %>%
  summarise(
    n_per_dom = n()
  )

# replace n_per_dom with one
URL1.allcountry.keep$n_per_dom <- 1
URL1.allcountry.keep <- URL1.allcountry.keep[!is.na(URL1.allcountry.keep$domain),]

# calculate again how many. now, its a count of how many diff participants per domain (because n_per_dom==1)
# delete if smaller/equal  50
URL1.allcountry.keep <- URL1.allcountry.keep %>% 
  group_by(domain) %>% 
  summarise(
    n_users = n()
  ) %>% 
  filter(n_users>=50) %>% 
  select(domain)
length(unique(URL1.allcountry.keep$domain))

# df withoout domains visited by fewer than 3 diff people
URL1.allcountry.reduced <- URL1.allcountry[URL1.allcountry$domain %in% URL1.allcountry.keep$domain, ]
rm(URL1.allcountry.keep)

# df to remove those visited fewer than 1200 times and less than 300 seconds
# basis is already reduced data set URL1.uk.reduced
URL1.allcountry.keep.2 <- URL1.allcountry.reduced %>% 
  group_by(domain) %>%
  summarise(
    d_per_dom = sum(active_seconds, na.rm = TRUE),
    n_per_dom = n()
  ) %>% 
  filter(d_per_dom >= 1200 & n_per_dom >= 300) %>% 
  ungroup() %>% 
  select(domain) # keep only domains

length(unique(URL1.allcountry.keep.2$domain))
# delete from reduced dataset
URL1.allcountry.reduced <- URL1.allcountry.reduced[URL1.allcountry.reduced$domain %in% URL1.allcountry.keep.2$domain, ]
rm(URL1.allcountry.keep.2)
## finally,  calculate duration and frequency per domain (only those that remain)
URL1.allcountry.dur <- URL1.allcountry.reduced %>% 
  group_by(panelist_id, domain) %>%
  summarise(
    d_per_dom = sum(active_seconds, na.rm = TRUE),
    n_per_dom = n()
  ) %>%
  ungroup() 
# number of domains
length(unique(URL1.allcountry.dur$domain))
# number of participants
length(unique(URL1.allcountry.dur$panelist_id))

weird.ones.allcountry <- URL1.allcountry[(!(URL1.allcountry$panelist_id %in% URL1.allcountry.dur$panelist_id)), ]

##################################################################################################################
#### for each country, create vars containing duration and frequency for most visited domains

## UK
URL1.uk <- URL1 %>%
  filter(country=="UK")
#how many participatns?
length(unique(URL1.uk$panelist_id))
#how many different domains?
length(unique(URL1.uk$domain))

### Limit to sites used by at least 3 different people
# calculate number of visits to each domain by each respondent
URL1.uk.keep <- URL1.uk %>% 
  group_by(domain, panelist_id) %>%
  summarise(
    n_per_dom = n()
  )

# replace n_per_dom with one
URL1.uk.keep$n_per_dom <- 1
URL1.uk.keep <- URL1.uk.keep[!is.na(URL1.uk.keep$domain),]

length((unique(URL1.uk.keep$domain)))
length((unique(URL1.uk.keep$panelist_id)))

# calculate again how many. now, its a count of how many diff participants per domain (because n_per_dom==1)
# delete if smaller/equal  3
URL1.uk.keep <- URL1.uk.keep %>% 
  group_by(domain) %>% 
  summarise(
    n_users = n()
  ) %>% 
  filter(n_users>=50) %>% 
  select(domain)
length(unique(URL1.uk.keep$domain))

# df withoout domains visited by fewer than 3 diff people
URL1.uk.reduced <- URL1.uk[URL1.uk$domain %in% URL1.uk.keep$domain, ]
rm(URL1.uk.keep)

# df to remove those visited fewer than 20 times and less than 60 seconds
# basis is already reduced data set URL1.uk.reduced
URL1.uk.keep.2 <- URL1.uk.reduced %>% 
  group_by(domain) %>%
  summarise(
    d_per_dom = sum(active_seconds, na.rm = TRUE),
    n_per_dom = n()
  ) %>% 
  filter(d_per_dom >= 1200 & n_per_dom >= 300) %>% 
  ungroup() %>% 
  select(domain) # keep only domains

length(unique(URL1.uk.keep.2$domain))
# delete from reduced dataset
URL1.uk.reduced <- URL1.uk.reduced[URL1.uk.reduced$domain %in% URL1.uk.keep.2$domain, ]
rm(URL1.uk.keep.2)
## finally,  calculate duration and frequency per domain (only those that remain)
URL1.uk.dur <- URL1.uk.reduced %>% 
  group_by(panelist_id, domain) %>%
  summarise(
    d_per_dom = sum(active_seconds, na.rm = TRUE),
    n_per_dom = n()
  ) %>%
  ungroup() 
# number of domains
length(unique(URL1.uk.dur$domain))
# number of participants
length(unique(URL1.uk.dur$panelist_id))

weird.ones.uk <- URL1.uk[(!(URL1.uk$panelist_id %in% URL1.uk.dur$panelist_id)), ]


## France
URL1.fr <- URL1 %>%
  filter(country=="France")
#how many participatns?
length(unique(URL1.fr$panelist_id))
#how many different domains?
length(unique(URL1.fr$domain))

### Limit to sites used by at least 3 different people
# calculate number of visits to each domain by each respondent
URL1.fr.keep <- URL1.fr %>% 
  group_by(domain, panelist_id) %>%
  summarise(
    n_per_dom = n()
  )

# replace n_per_dom with one
URL1.fr.keep$n_per_dom <- 1
URL1.fr.keep <- URL1.fr.keep[!is.na(URL1.fr.keep$domain),]

length((unique(URL1.fr.keep$domain)))
length((unique(URL1.fr.keep$panelist_id)))

# calculate again how many. now, its a count of how many diff participants per domain (because n_per_dom==1)
# delete if smaller/equal  3
URL1.fr.keep <- URL1.fr.keep %>% 
  group_by(domain) %>% 
  summarise(
    n_users = n()
  ) %>% 
  filter(n_users>=50) %>% 
  select(domain)
length(unique(URL1.fr.keep$domain))

# df withoout domains visited by fewer than 3 diff people
URL1.fr.reduced <- URL1.fr[URL1.fr$domain %in% URL1.fr.keep$domain, ]
rm(URL1.fr.keep)

# df to remove those visited fewer than 20 times and less than 60 seconds
# basis is already reduced data set URL1.fr.reduced
URL1.fr.keep.2 <- URL1.fr.reduced %>% 
  group_by(domain) %>%
  summarise(
    d_per_dom = sum(active_seconds, na.rm = TRUE),
    n_per_dom = n()
  ) %>% 
  filter(d_per_dom >= 1200 & n_per_dom >= 300) %>% 
  ungroup() %>% 
  select(domain) # keep only domains

length(unique(URL1.fr.keep.2$domain))
# delete from reduced dataset
URL1.fr.reduced <- URL1.fr.reduced[URL1.fr.reduced$domain %in% URL1.fr.keep.2$domain, ]
rm(URL1.fr.keep.2)
## finally,  calculate duration and frequency per domain (only those that remain)
URL1.fr.dur <- URL1.fr.reduced %>% 
  group_by(panelist_id, domain) %>%
  summarise(
    d_per_dom = sum(active_seconds, na.rm = TRUE),
    n_per_dom = n()
  ) %>%
  ungroup() 
# number of domains
length(unique(URL1.fr.dur$domain))
# number of participants
length(unique(URL1.fr.dur$panelist_id))

weird.ones.fr <- URL1.fr[(!(URL1.fr$panelist_id %in% URL1.fr.dur$panelist_id)), ]


## Germany
URL1.de <- URL1 %>%
  filter(country=="Germany")
#how many participatns?
length(unique(URL1.de$panelist_id))
#how many different domains?
length(unique(URL1.de$domain))

### Limit to sites used by at least 3 different people
# calculate number of visits to each domain by each respondent
URL1.de.keep <- URL1.de %>% 
  group_by(domain, panelist_id) %>%
  summarise(
    n_per_dom = n()
  )

# replace n_per_dom with one
URL1.de.keep$n_per_dom <- 1
URL1.de.keep <- URL1.de.keep[!is.na(URL1.de.keep$domain),]

length((unique(URL1.de.keep$domain)))
length((unique(URL1.de.keep$panelist_id)))

# calculate again how many. now, its a count of how many diff participants per domain (because n_per_dom==1)
# delete if smaller/equal  3
URL1.de.keep <- URL1.de.keep %>% 
  group_by(domain) %>% 
  summarise(
    n_users = n()
  ) %>% 
  filter(n_users>=50) %>% 
  select(domain)
length(unique(URL1.de.keep$domain))

# df withoout domains visited by fewer than 3 diff people
URL1.de.reduced <- URL1.de[URL1.de$domain %in% URL1.de.keep$domain, ]
rm(URL1.de.keep)

# df to remove those visited fewer than 20 times and less than 60 seconds
# basis is already reduced data set URL1.de.reduced
URL1.de.keep.2 <- URL1.de.reduced %>% 
  group_by(domain) %>%
  summarise(
    d_per_dom = sum(active_seconds, na.rm = TRUE),
    n_per_dom = n()
  ) %>% 
  filter(d_per_dom >= 1200 & n_per_dom >= 300) %>% 
  ungroup() %>% 
  select(domain) # keep only domains

length(unique(URL1.de.keep.2$domain))
# delete from reduced dataset
URL1.de.reduced <- URL1.de.reduced[URL1.de.reduced$domain %in% URL1.de.keep.2$domain, ]
rm(URL1.de.keep.2)
## finally,  calculate duration and frequency per domain (only those that remain)
URL1.de.dur <- URL1.de.reduced %>% 
  group_by(panelist_id, domain) %>%
  summarise(
    d_per_dom = sum(active_seconds, na.rm = TRUE),
    n_per_dom = n()
  ) %>%
  ungroup() 
# number of domains
length(unique(URL1.de.dur$domain))
# number of participants
length(unique(URL1.de.dur$panelist_id))

weird.ones.de <- URL1.de[(!(URL1.de$panelist_id %in% URL1.de.dur$panelist_id)), ]



### #remove what is obsolete
rm(URL1.de.reduced,URL1.fr.reduced,URL1.uk.reduced)



#######calculate time and frequency per domain
overlap <- URL1.de.dur[URL1.de.dur$domain %in% URL1.fr.dur$domain,]
overlap <- overlap[overlap$domain %in% URL1.uk.dur$domain,]
overlap <- overlap %>% 
  distinct(domain) %>% 
  select(domain)

length(overlap$domain) # only 66 overlapping domains between countries

#######calculate time and frequency per app

# ALL
URL1.allcountry.dur <- URL1.allcountry.dur[URL1.allcountry.dur$domain %in% overlap$domain,]
dur.all <- select(URL1.allcountry.dur, -n_per_dom)
fre.all <- select(URL1.allcountry.dur, -d_per_dom)


dur.all <- spread(dur.all, domain, d_per_dom)
fre.all <- spread(fre.all, domain, n_per_dom)

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
dur.de <- select(URL1.de.dur, -n_per_dom)
fre.de <- select(URL1.de.dur, -d_per_dom)


dur.de <- spread(dur.de, domain, d_per_dom)
fre.de <- spread(fre.de, domain, n_per_dom)

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
dur.fr <- select(URL1.fr.dur, -n_per_dom)
fre.fr <- select(URL1.fr.dur, -d_per_dom)


dur.fr <- spread(dur.fr, domain, d_per_dom)
fre.fr <- spread(fre.fr, domain, n_per_dom)

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
dur.uk <- select(URL1.uk.dur, -n_per_dom)
fre.uk <- select(URL1.uk.dur, -d_per_dom)


dur.uk <- spread(dur.uk, domain, d_per_dom)
fre.uk <- spread(fre.uk, domain, n_per_dom)

names.dur.uk <- names(dur.uk)
names.fre.uk <- names(fre.uk)

names.dur.uk <- paste0('d_' , names.dur.uk)
names.fre.uk <- paste0('n_' , names.fre.uk)

dur.uk <- set_names(dur.uk, nm = names.dur.uk)
fre.uk <- set_names(fre.uk, nm = names.fre.uk)
rm(names.dur.uk,names.fre.uk)


dur.uk <- rename(dur.uk, panelist_id = d_panelist_id)
fre.uk <- rename(fre.uk, panelist_id = n_panelist_id)



saveRDS(dur.de, "./data/work/dur_dom_de.RDS")
saveRDS(fre.de, "./data/work/fre_dom_de.RDS")

saveRDS(dur.fr, "./data/work/dur_dom_fr.RDS")
saveRDS(fre.fr, "./data/work/fre_dom_fr.RDS")

saveRDS(dur.uk, "./data/work/dur_dom_uk.RDS")
saveRDS(fre.uk, "./data/work/fre_dom_uk.RDS")

saveRDS(dur.all, "./data/work/dur_dom_all.RDS")
saveRDS(fre.all, "./data/work/fre_dom_all.RDS")
