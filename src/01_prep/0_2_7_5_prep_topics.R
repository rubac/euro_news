# setwd("/home/wrszemsrgyercqh/an_work/EU/")

library(tidyverse)

###### Load topic data

lda <- read.csv("all_lda_topics.csv", stringsAsFactors = FALSE)

nmf <- read.csv("all_nmf_topics.csv", stringsAsFactors = FALSE)

###### aggregate LDA topics on respondent level

lda.g <- lda %>% 
  group_by(pseudonym, topic_english) %>% 
  dplyr::summarise(
    n_per_lda = n(),
    d_per_lda = sum(duration, na.rm = TRUE))

lda.n <- spread(lda.g, topic_english, n_per_lda)
lda.d <- spread(lda.g, topic_english, d_per_lda)

lda.gn <- lda.n %>%
  group_by(pseudonym) %>% 
  dplyr::summarise(
    lda1_n = sum(`0`, na.rm = TRUE),
    lda2_n = sum(`1`, na.rm = TRUE),
    lda3_n = sum(`2`, na.rm = TRUE),
    lda4_n = sum(`3`, na.rm = TRUE),
    lda5_n = sum(`4`, na.rm = TRUE),
    lda6_n = sum(`5`, na.rm = TRUE),
    lda7_n = sum(`6`, na.rm = TRUE),
    lda8_n = sum(`7`, na.rm = TRUE),
    lda9_n = sum(`8`, na.rm = TRUE),
    lda10_n = sum(`9`, na.rm = TRUE),
    lda11_n = sum(`10`, na.rm = TRUE),
    lda12_n = sum(`11`, na.rm = TRUE),
    lda13_n = sum(`12`, na.rm = TRUE),
    lda14_n = sum(`13`, na.rm = TRUE),
    lda15_n = sum(`14`, na.rm = TRUE),
    lda16_n = sum(`15`, na.rm = TRUE),
    lda17_n = sum(`16`, na.rm = TRUE),
    lda18_n = sum(`17`, na.rm = TRUE),
    lda19_n = sum(`18`, na.rm = TRUE),
    lda20_n = sum(`19`, na.rm = TRUE))

lda.gd <- lda.d %>%
  group_by(pseudonym) %>% 
  dplyr::summarise(
    lda1_d = sum(`0`, na.rm = TRUE),
    lda2_d = sum(`1`, na.rm = TRUE),
    lda3_d = sum(`2`, na.rm = TRUE),
    lda4_d = sum(`3`, na.rm = TRUE),
    lda5_d = sum(`4`, na.rm = TRUE),
    lda6_d = sum(`5`, na.rm = TRUE),
    lda7_d = sum(`6`, na.rm = TRUE),
    lda8_d = sum(`7`, na.rm = TRUE),
    lda9_d = sum(`8`, na.rm = TRUE),
    lda10_d = sum(`9`, na.rm = TRUE),
    lda11_d = sum(`10`, na.rm = TRUE),
    lda12_d = sum(`11`, na.rm = TRUE),
    lda13_d = sum(`12`, na.rm = TRUE),
    lda14_d = sum(`13`, na.rm = TRUE),
    lda15_d = sum(`14`, na.rm = TRUE),
    lda16_d = sum(`15`, na.rm = TRUE),
    lda17_d = sum(`16`, na.rm = TRUE),
    lda18_d = sum(`17`, na.rm = TRUE),
    lda19_d = sum(`18`, na.rm = TRUE),
    lda20_d = sum(`19`, na.rm = TRUE))

###### aggregate NMF topics on respondent level

nmf.g <- nmf %>% 
  group_by(pseudonym, topic_english) %>% 
  dplyr::summarise(
    n_per_nmf = n(),
    d_per_nmf = sum(duration, na.rm = TRUE))

nmf.n <- spread(nmf.g, topic_english, n_per_nmf)
nmf.d <- spread(nmf.g, topic_english, d_per_nmf)

nmf.gn <- nmf.n %>%
  group_by(pseudonym) %>% 
  dplyr::summarise(
    nmf1_n = sum(`0`, na.rm = TRUE),
    nmf2_n = sum(`1`, na.rm = TRUE),
    nmf3_n = sum(`2`, na.rm = TRUE),
    nmf4_n = sum(`3`, na.rm = TRUE),
    nmf5_n = sum(`4`, na.rm = TRUE),
    nmf6_n = sum(`5`, na.rm = TRUE),
    nmf7_n = sum(`6`, na.rm = TRUE),
    nmf8_n = sum(`7`, na.rm = TRUE),
    nmf9_n = sum(`8`, na.rm = TRUE),
    nmf10_n = sum(`9`, na.rm = TRUE),
    nmf11_n = sum(`10`, na.rm = TRUE),
    nmf12_n = sum(`11`, na.rm = TRUE),
    nmf13_n = sum(`12`, na.rm = TRUE),
    nmf14_n = sum(`13`, na.rm = TRUE),
    nmf15_n = sum(`14`, na.rm = TRUE),
    nmf16_n = sum(`15`, na.rm = TRUE),
    nmf17_n = sum(`16`, na.rm = TRUE),
    nmf18_n = sum(`17`, na.rm = TRUE),
    nmf19_n = sum(`18`, na.rm = TRUE),
    nmf20_n = sum(`19`, na.rm = TRUE))

nmf.gd <- nmf.d %>%
  group_by(pseudonym) %>% 
  dplyr::summarise(
    nmf1_d = sum(`0`, na.rm = TRUE),
    nmf2_d = sum(`1`, na.rm = TRUE),
    nmf3_d = sum(`2`, na.rm = TRUE),
    nmf4_d = sum(`3`, na.rm = TRUE),
    nmf5_d = sum(`4`, na.rm = TRUE),
    nmf6_d = sum(`5`, na.rm = TRUE),
    nmf7_d = sum(`6`, na.rm = TRUE),
    nmf8_d = sum(`7`, na.rm = TRUE),
    nmf9_d = sum(`8`, na.rm = TRUE),
    nmf10_d = sum(`9`, na.rm = TRUE),
    nmf11_d = sum(`10`, na.rm = TRUE),
    nmf12_d = sum(`11`, na.rm = TRUE),
    nmf13_d = sum(`12`, na.rm = TRUE),
    nmf14_d = sum(`13`, na.rm = TRUE),
    nmf15_d = sum(`14`, na.rm = TRUE),
    nmf16_d = sum(`15`, na.rm = TRUE),
    nmf17_d = sum(`16`, na.rm = TRUE),
    nmf18_d = sum(`17`, na.rm = TRUE),
    nmf19_d = sum(`18`, na.rm = TRUE),
    nmf20_d = sum(`19`, na.rm = TRUE))

####################
### Merge and save

topics <- 
  lda.gn %>%
  left_join(lda.gd, by = "pseudonym") %>%
  left_join(nmf.gn, by = "pseudonym") %>%
  left_join(nmf.gd, by = "pseudonym")

saveRDS(topics, "./data/work/topics_all.RDS")
