##################################################################################

library(tidyverse)
library(ggcorrplot)
library(rtf)
library(xtable)

# Set path
# setwd("/home/r_uma_2019/respondi_eu/")

##################################################################################
# BERT Features - Stats and Plots
##################################################################################

# load data
load("./src/02_analysis/prep_predict_all.Rdata")

# Means overall

track_de_c <- track_de %>% filter(no_reduced_bert == "reduced_bert")
track_fr_c <- track_fr %>% filter(no_reduced_bert == "reduced_bert")
track_uk_c <- track_uk %>% filter(no_reduced_bert == "reduced_bert")

sub1_de <- track_de %>%
  mutate(d_polinterest = as.numeric(polinterest) - 1,
         d_undecided = as.numeric(undecided) - 1,
         d_voted = as.numeric(voted) - 1,
         d_changed = as.numeric(changed) - 1) %>%
  summarise(min(d_polinterest, na.rm = T), mean(d_polinterest, na.rm = T), max(d_polinterest, na.rm = T), sum(!is.na(d_polinterest)), 
            min(d_undecided, na.rm = T), mean(d_undecided, na.rm = T), max(d_undecided, na.rm = T), sum(!is.na(d_undecided)),  
            min(d_voted, na.rm = T), mean(d_voted, na.rm = T), max(d_voted, na.rm = T), sum(!is.na(d_voted)), 
            min(d_changed, na.rm = T), mean(d_changed, na.rm = T), max(d_changed, na.rm = T), sum(!is.na(d_changed)))

sub2_de <- track_de_c %>%
  summarise(min(clus10_n_r, na.rm = T), mean(clus10_n_r, na.rm = T), max(clus10_n_r, na.rm = T), sum(!is.na(clus10_n_r)), 
            min(clus16_n_r, na.rm = T), mean(clus16_n_r, na.rm = T), max(clus16_n_r, na.rm = T), sum(!is.na(clus16_n_r)),  
            min(clus23_n_r, na.rm = T), mean(clus23_n_r, na.rm = T), max(clus23_n_r, na.rm = T), sum(!is.na(clus23_n_r)), 
            min(median_dim2_r, na.rm = T), mean(median_dim2_r, na.rm = T), max(median_dim2_r, na.rm = T), sum(!is.na(median_dim2_r)),
            min(var_dim2_r, na.rm = T), mean(var_dim2_r, na.rm = T), max(var_dim2_r, na.rm = T), sum(!is.na(var_dim2_r)))

m1 <- matrix(sub1_de, ncol = 4, byrow = T)
m2 <- matrix(sub2_de, ncol = 4, byrow = T)
tab1 <- rbind(m1, m2)
tab <- xtable(tab1)
print(tab, type = "latex", file = "desc_de.tex")

sub1_fr <- track_fr %>%
  mutate(d_polinterest = as.numeric(polinterest) - 1,
         d_undecided = as.numeric(undecided) - 1,
         d_voted = as.numeric(voted) - 1,
         d_changed = as.numeric(changed) - 1) %>%
  summarise(min(d_polinterest, na.rm = T), mean(d_polinterest, na.rm = T), max(d_polinterest, na.rm = T), sum(!is.na(d_polinterest)), 
            min(d_undecided, na.rm = T), mean(d_undecided, na.rm = T), max(d_undecided, na.rm = T), sum(!is.na(d_undecided)),  
            min(d_voted, na.rm = T), mean(d_voted, na.rm = T), max(d_voted, na.rm = T), sum(!is.na(d_voted)), 
            min(d_changed, na.rm = T), mean(d_changed, na.rm = T), max(d_changed, na.rm = T), sum(!is.na(d_changed)))

sub2_fr <- track_fr_c %>%
  summarise(min(clus10_n_r, na.rm = T), mean(clus10_n_r, na.rm = T), max(clus10_n_r, na.rm = T), sum(!is.na(clus10_n_r)), 
            min(clus16_n_r, na.rm = T), mean(clus16_n_r, na.rm = T), max(clus16_n_r, na.rm = T), sum(!is.na(clus16_n_r)),  
            min(clus23_n_r, na.rm = T), mean(clus23_n_r, na.rm = T), max(clus23_n_r, na.rm = T), sum(!is.na(clus23_n_r)), 
            min(median_dim2_r, na.rm = T), mean(median_dim2_r, na.rm = T), max(median_dim2_r, na.rm = T), sum(!is.na(median_dim2_r)),
            min(var_dim2_r, na.rm = T), mean(var_dim2_r, na.rm = T), max(var_dim2_r, na.rm = T), sum(!is.na(var_dim2_r)))

m1 <- matrix(sub1_fr, ncol = 4, byrow = T)
m2 <- matrix(sub2_fr, ncol = 4, byrow = T)
tab2 <- rbind(m1, m2)
tab <- xtable(tab2)
print(tab, type = "latex", file = "desc_fr.tex")

sub1_uk <- track_uk %>%
  mutate(d_polinterest = as.numeric(polinterest) - 1,
         d_undecided = as.numeric(undecided) - 1,
         d_voted = as.numeric(voted) - 1,
         d_changed = as.numeric(changed) - 1) %>%
  summarise(min(d_polinterest, na.rm = T), mean(d_polinterest, na.rm = T), max(d_polinterest, na.rm = T), sum(!is.na(d_polinterest)), 
            min(d_undecided, na.rm = T), mean(d_undecided, na.rm = T), max(d_undecided, na.rm = T), sum(!is.na(d_undecided)),  
            min(d_voted, na.rm = T), mean(d_voted, na.rm = T), max(d_voted, na.rm = T), sum(!is.na(d_voted)), 
            min(d_changed, na.rm = T), mean(d_changed, na.rm = T), max(d_changed, na.rm = T), sum(!is.na(d_changed)))

sub2_uk <- track_uk_c %>%
  summarise(min(clus10_n_r, na.rm = T), mean(clus10_n_r, na.rm = T), max(clus10_n_r, na.rm = T), sum(!is.na(clus10_n_r)), 
            min(clus16_n_r, na.rm = T), mean(clus16_n_r, na.rm = T), max(clus16_n_r, na.rm = T), sum(!is.na(clus16_n_r)),  
            min(clus23_n_r, na.rm = T), mean(clus23_n_r, na.rm = T), max(clus23_n_r, na.rm = T), sum(!is.na(clus23_n_r)), 
            min(median_dim2_r, na.rm = T), mean(median_dim2_r, na.rm = T), max(median_dim2_r, na.rm = T), sum(!is.na(median_dim2_r)),
            min(var_dim2_r, na.rm = T), mean(var_dim2_r, na.rm = T), max(var_dim2_r, na.rm = T), sum(!is.na(var_dim2_r)))

m1 <- matrix(sub1_uk, ncol = 4, byrow = T)
m2 <- matrix(sub2_uk, ncol = 4, byrow = T)
tab3 <- rbind(m1, m2)
tab <- xtable(tab3)
print(tab, type = "latex", file = "desc_uk.tex")

# Means by group

sub1_de <- track_de_c %>%
  select(undecided, voted, changed, polinterest, clus10_n_r, clus16_n_r, clus23_n_r, median_dim2_r, var_dim2_r) %>%
  group_by(undecided) %>%
  summarise(mean(clus10_n_r), mean(clus16_n_r), mean(clus23_n_r), mean(median_dim2_r), mean(var_dim2_r)) %>%
  rename(group = undecided)

sub2_de <- track_de_c %>%
  select(undecided, voted, changed, polinterest, clus10_n_r, clus16_n_r, clus23_n_r, median_dim2_r, var_dim2_r) %>%
  group_by(voted) %>%
  summarise(mean(clus10_n_r), mean(clus16_n_r), mean(clus23_n_r), mean(median_dim2_r), mean(var_dim2_r)) %>%
  rename(group = voted)

sub3_de <- track_de_c %>%
  select(undecided, voted, changed, polinterest, clus10_n_r, clus16_n_r, clus23_n_r, median_dim2_r, var_dim2_r) %>%
  group_by(changed) %>%
  summarise(mean(clus10_n_r), mean(clus16_n_r), mean(clus23_n_r), mean(median_dim2_r), mean(var_dim2_r)) %>%
  rename(group = changed) %>% drop_na() 

sub4_de <- track_de_c %>%
  select(undecided, voted, changed, polinterest, clus10_n_r, clus16_n_r, clus23_n_r, median_dim2_r, var_dim2_r) %>%
  group_by(polinterest) %>%
  summarise(mean(clus10_n_r), mean(clus16_n_r), mean(clus23_n_r), mean(median_dim2_r), mean(var_dim2_r)) %>%
  rename(group = polinterest)

tab1 <- rbind(sub4_de[,1:4], sub1_de[,1:4], sub2_de[,1:4], sub3_de[,1:4])
tab1 <- t(cbind(tab1[,1], round(tab1[,2:4], digits = 2)))

tab <- xtable(tab1)
print(tab, type = "latex", file = "cluster_de.tex")

tab2 <- rbind(sub4_de[,c(1,5,6)], sub1_de[,c(1,5,6)], sub2_de[,c(1,5,6)], sub3_de[,c(1,5,6)])
tab2 <- t(cbind(tab2[,1], round(tab2[,2:3], digits = 2)))

tab <- xtable(tab2)
print(tab, type = "latex", file = "PCA_de.tex")

sub1_fr <- track_fr_c %>%
  select(undecided, voted, changed, polinterest, clus10_n_r, clus16_n_r, clus23_n_r, median_dim2_r, var_dim2_r) %>%
  group_by(undecided) %>%
  summarise(mean(clus10_n_r), mean(clus16_n_r), mean(clus23_n_r), mean(median_dim2_r), mean(var_dim2_r)) %>%
  rename(group = undecided)

sub2_fr <- track_fr_c %>%
  select(undecided, voted, changed, polinterest, clus10_n_r, clus16_n_r, clus23_n_r, median_dim2_r, var_dim2_r) %>%
  group_by(voted) %>%
  summarise(mean(clus10_n_r), mean(clus16_n_r), mean(clus23_n_r), mean(median_dim2_r), mean(var_dim2_r)) %>%
  rename(group = voted)

sub3_fr <- track_fr_c %>%
  select(undecided, voted, changed, polinterest, clus10_n_r, clus16_n_r, clus23_n_r, median_dim2_r, var_dim2_r) %>%
  group_by(changed) %>%
  summarise(mean(clus10_n_r), mean(clus16_n_r), mean(clus23_n_r), mean(median_dim2_r), mean(var_dim2_r)) %>%
  rename(group = changed) %>% drop_na()  

sub4_fr <- track_fr_c %>%
  select(undecided, voted, changed, polinterest, clus10_n_r, clus16_n_r, clus23_n_r, median_dim2_r, var_dim2_r) %>%
  group_by(polinterest) %>%
  summarise(mean(clus10_n_r), mean(clus16_n_r), mean(clus23_n_r), mean(median_dim2_r), mean(var_dim2_r)) %>%
  rename(group = polinterest) 

tab1 <- rbind(sub4_fr[,1:4], sub1_fr[,1:4], sub2_fr[,1:4], sub3_fr[,1:4])
tab1 <- t(cbind(tab1[,1], round(tab1[,2:4], digits = 2)))

tab <- xtable(tab1)
print(tab, type = "latex", file = "cluster_fr.tex")

tab2 <- rbind(sub4_fr[,c(1,5,6)], sub1_fr[,c(1,5,6)], sub2_fr[,c(1,5,6)], sub3_fr[,c(1,5,6)])
tab2 <- t(cbind(tab2[,1], round(tab2[,2:3], digits = 2)))

tab <- xtable(tab2)
print(tab, type = "latex", file = "PCA_fr.tex")

sub1_uk <- track_uk_c %>%
  select(undecided, voted, changed, polinterest, clus10_n_r, clus16_n_r, clus23_n_r, median_dim2_r, var_dim2_r) %>%
  group_by(undecided) %>%
  summarise(mean(clus10_n_r), mean(clus16_n_r), mean(clus23_n_r), mean(median_dim2_r), mean(var_dim2_r)) %>%
  rename(group = undecided)

sub2_uk <- track_uk_c %>%
  select(undecided, voted, changed, polinterest, clus10_n_r, clus16_n_r, clus23_n_r, median_dim2_r, var_dim2_r) %>%
  group_by(voted) %>%
  summarise(mean(clus10_n_r), mean(clus16_n_r), mean(clus23_n_r), mean(median_dim2_r), mean(var_dim2_r)) %>%
  rename(group = voted)

sub3_uk <- track_uk_c %>%
  select(undecided, voted, changed, polinterest, clus10_n_r, clus16_n_r, clus23_n_r, median_dim2_r, var_dim2_r) %>%
  group_by(changed) %>%
  summarise(mean(clus10_n_r), mean(clus16_n_r), mean(clus23_n_r), mean(median_dim2_r), mean(var_dim2_r)) %>%
  rename(group = changed) %>% drop_na() 

sub4_uk <- track_uk_c %>%
  select(undecided, voted, changed, polinterest, clus10_n_r, clus16_n_r, clus23_n_r, median_dim2_r, var_dim2_r) %>%
  group_by(polinterest) %>%
  summarise(mean(clus10_n_r), mean(clus16_n_r), mean(clus23_n_r), mean(median_dim2_r), mean(var_dim2_r)) %>%
  rename(group = polinterest)

tab1 <- rbind(sub4_uk[,1:4], sub1_uk[,1:4], sub2_uk[,1:4], sub3_uk[,1:4])
tab1 <- t(cbind(tab1[,1], round(tab1[,2:4], digits = 2)))

tab <- xtable(tab1)
print(tab, type = "latex", file = "cluster_uk.tex")

tab2 <- rbind(sub4_uk[,c(1,5,6)], sub1_uk[,c(1,5,6)], sub2_uk[,c(1,5,6)], sub3_uk[,c(1,5,6)])
tab2 <- t(cbind(tab2[,1], round(tab2[,2:3], digits = 2)))

tab <- xtable(tab2)
print(tab, type = "latex", file = "PCA_uk.tex")

# Corrplots

track_de_train_prep <- 
  track_de_train %>%
  filter(no_reduced_bert == "reduced_bert") %>%
  select(undecided, voted, changed, polinterest,
         "Cluster 2 n" = clus2_n_r, "Cluster 19 n" = clus19_n_r, "Cluster 25 n" = clus25_n_r, "Cluster 27 n" = clus27_n_r) %>%
  mutate(undecided = ifelse(undecided == "undecided", 1, 0),
         voted = ifelse(voted == "Yes", 1, 0),
         changed = ifelse(changed == "changed", 1, 0),
         polinterest = ifelse(polinterest == "high", 1, 0))

corrs_de <- cor(track_de_train_prep, use = "pairwise.complete.obs")

ggcorrplot(corrs_de, type = "lower",
           lab = TRUE,  
           outline.col = "white",
           colors = c("red", "snow2", "snow4"))
ggsave("de_corrplot.png", width = 8, height = 8)

track_fr_train_prep <- 
  track_fr_train %>%
  filter(no_reduced_bert == "reduced_bert") %>%
  select(undecided, voted, changed, polinterest,
         "Cluster 2 n" = clus2_n_r, "Cluster 19 n" = clus19_n_r, "Cluster 25 n" = clus25_n_r, "Cluster 27 n" = clus27_n_r) %>%
  mutate(undecided = ifelse(undecided == "undecided", 1, 0),
         voted = ifelse(voted == "Yes", 1, 0),
         changed = ifelse(changed == "changed", 1, 0),
         polinterest = ifelse(polinterest == "high", 1, 0))

corrs_fr <- cor(track_fr_train_prep, use = "pairwise.complete.obs")

ggcorrplot(corrs_fr, type = "lower",
           lab = TRUE,  
           outline.col = "white",
           colors = c("red", "snow2", "snow4"))
ggsave("fr_corrplot.png", width = 8, height = 8)

track_uk_train_prep <- 
  track_uk_train %>%
  filter(no_reduced_bert == "reduced_bert") %>%
  select(undecided, voted, changed, polinterest,
         "Cluster 2 n" = clus2_n_r, "Cluster 19 n" = clus19_n_r, "Cluster 25 n" = clus25_n_r, "Cluster 27 n" = clus27_n_r) %>%
  mutate(undecided = ifelse(undecided == "undecided", 1, 0),
         voted = ifelse(voted == "Yes", 1, 0),
         changed = ifelse(changed == "changed", 1, 0),
         polinterest = ifelse(polinterest == "high", 1, 0))

corrs_uk <- cor(track_uk_train_prep, use = "pairwise.complete.obs")

ggcorrplot(corrs_uk, type = "lower",
           lab = TRUE,  
           outline.col = "white",
           colors = c("red", "snow2", "snow4"))
ggsave("uk_corrplot.png", width = 8, height = 8)
