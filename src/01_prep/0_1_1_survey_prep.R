setwd("/home/wrszemsrgyercqh/an_work/EU/")

library(tidyverse)
library(haven)

# Survey data data prep code unfortunaley overwritten,
# but here is what was done to get from sdata.RData to dat_surv.RDS 
# See questionnaire for details on the quesions asked
#   - Panelist_id: Unique person identifier
#   - Country: Identifies a user's country
#   - reg_vote: Whether the user is registered to vote (v_8)
#   - tv_hh: Whether there is a TV in the hh (v_9)
#   - has.twitter: Whether the user has a twitter account (v_10-14)
#   - has.facebook: Whether the user has a facebook account (v_10-14)
#   - has.instagram: Whether the user has an instagram account (v_10-14)
#   - has.linkedin: Whether the user has a linkedin account (v_10-14)
#   - has.oth.smedia: Whether the user has additional social media accounts (v_10-14)
#   - linkedin: Whether the user intends to vote in the upcoming election (v_15)
#   - fr.pres.vote: Whether the user voted for Le Pen or Macron in 2017 French election (2nd round) (v_20)
#   - fr.yellow.v.supp: Whether the user supports (high) the yellow vest protests or not (low) (v_21)
#   - uk.2015.vote: Which party the user voted for in 2015 UK election (v_30)
#   - uk.brexit.vote: Whether the user voted for or against Brexit in referendum (v_31)
#   - de.wahlomat.use: Whether the user used Wahlomat in EU elections (DE only) (v_45)
#   - de.wahlomat.pty: Which party Wahlomat suggested (DE only) (v_46)
#   - voted: Whether the user voted in the EU election (v_9_2)
#   - change: Whether the user changed her mind about her choice (actual vote different from intention)
#   - intent_pty_cert: How sure the user is in his intended vote (v_17/v_27/v_38)
#   - undecided: Whether the user is undecided which party to vote for (v_16/v_26/v_37)
#   - v.pty.intent: Which party the user intends to vote (v_16/v_26/v_37)
#   - polinterest: Whether the user is interested in politics or not (v_25/v_36/v_44)
#   - polinterest.num: How much the user is interested in politics or not (v_25/v_36/v_44)
#   - leftmidright: Whether the user leans left, middle, or right (v_24/v_35/v_43)
#   - leftmidright.num: How much the user leans left, middle, or right (v_24/v_35/v_43)
#   - pty.feel.close: Which party the user feels closest to (v_18/v_28/v_39)
#   - last.election: Which party the user voted for in last national election (v_19/v_29/v_40)
#   - trust.EP: How much the user trusts the European parliament (v_22/v_32/v_41)
#   - trust.nat.pol: How much the user trusts the national parliament/other governmental institutions (v_23/v_33/v_42)

df.survey <- readRDS("./data/work/dat_surv.rds")

# In addition, panel.RData, panel_bis.RData and panel_bis2.sav were
# combined into a single file with as much sociodemographic info as possible

load("./data/orig/panel.RData")

panel_sub <- select(panel, pseudonym, gender, age, children)

load("./data/orig/panel_bis.RData")

panel_bis$region <- panel_bis$region_FR 
panel_bis$region <- ifelse(panel_bis$country == "16: United Kingdom", as.character(panel_bis$region_UK), as.character(panel_bis$region))
panel_bis$region <- ifelse(panel_bis$country == "9: Germany", as.character(panel_bis$region_DE), as.character(panel_bis$region))
panel_bis$region <- as.factor(panel_bis$region)

panel_bis$income <- panel_bis$income_FR_DE
panel_bis$income <- ifelse(panel_bis$country == "16: United Kingdom", as.character(panel_bis$income_UK), as.character(panel_bis$income))
panel_bis$income <- fct_recode(panel_bis$income, 
                               "1: under 500 EUR" = "1: Under 500 pounds",
                               "2: 500 to 1.000 EUR" = "2: 500 to 1,000 pounds",
                               "3: 1.000 to 1.500 EUR" = "3: 1,000 to 1,500 pounds",
                               "4: 1.500 to 2.000 EUR" = "4: 1,500 to 2,000 pounds",
                               "6: 2.500 to 3.000 EUR" = "5: 2,000 to 2,500 pounds",
                               "7: 3.000 and more EUR" = "6: Over 2,500 pounds",
                               "7: 3.000 and more EUR" = "7: 3.000 to 3.500 EUR",
                               "7: 3.000 and more EUR" = "8: 3.500 to 4.000 EUR",
                               "7: 3.000 and more EUR" = "9: 4.000 to 4.500 EUR",
                               "7: 3.000 and more EUR" = "10: 4.500 to 5.000 EUR",
                               "7: 3.000 and more EUR" = "11: more than 5.000 EUR",
                               "98: no income" = "98: No joint Household income -")
  
panel_bis$income_FR_DE <- fct_recode(panel_bis$income_FR_DE, 
                                     "1: under 1000 EUR" = "98: no income",
                                     "1: under 1000 EUR" = "1: under 500 EUR",
                                     "1: under 1000 EUR" = "2: 500 to 1.000 EUR",
                                     "9: more than 4.000 EUR" = "9: 4.000 to 4.500 EUR",
                                     "9: more than 4.000 EUR" = "10: 4.500 to 5.000 EUR",
                                     "9: more than 4.000 EUR" = "11: more than 5.000 EUR")

panel_bis$income_UK <- fct_recode(panel_bis$income_UK, 
                                  "1: Under 1000 pounds" = "98: No joint Household income -",
                                  "1: Under 1000 pounds" = "1: Under 500 pounds",
                                  "1: Under 1000 pounds" = "2: 500 to 1,000 pounds")

panel_bis$education <- panel_bis$education_DE
panel_bis$education <- ifelse(panel_bis$country == "16: United Kingdom", as.character(panel_bis$education_UK), as.character(panel_bis$education))
panel_bis$education <- ifelse(panel_bis$country == "12: France", as.character(panel_bis$education_FR), as.character(panel_bis$education))
panel_bis$education <- fct_recode(panel_bis$education,
                                  "No qualification (yet)" = "4: No qualification (yet)",
                                  "No qualification (yet)" = "NA: No qualification (yet)",
                                  "No qualification (yet)" = "98: No formal education or qualifications (yet)")

panel_bis$family <- panel_bis$family_UK_DE
panel_bis$family <- ifelse(panel_bis$country == "12: France", as.character(panel_bis$family_FR), as.character(panel_bis$family))
panel_bis$family <- fct_recode(panel_bis$family, 
                               "1: Married" = "1: Marié",
                               "2: Civil Partnership" = "2: Pacsé",
                               "3: Single, living with partner" = "3: En couple",
                               "4: Single, not living with partner" = "4: Célibataire",
                               "5: Divorced/widowed, living with partner" = "5: Divorcé/Veuf, en couple",
                               "6: Divorced/widowed, not living with partner" = "6: Divorcé/Veuf, célibataire")

panel_bis$family_FR <- fct_recode(panel_bis$family_FR, 
                               "1: Marié" = "2: Pacsé",
                               "2: En couple" = "3: En couple",
                               "2: En couple" = "4: Célibataire",
                               "4: Divorcé/Veuf" = "5: Divorcé/Veuf, en couple",
                               "4: Divorcé/Veuf" = "6: Divorcé/Veuf, célibataire")

panel_bis$family_UK_DE <-fct_recode(panel_bis$family_UK_DE, 
                                    "1: Married" = "2: Civil Partnership",
                                    "2: Single" = "3: Single, living with partner",
                                    "2: Single" = "4: Single, not living with partner",
                                    "3: Divorced/widowed" = "5: Divorced/widowed, living with partner",
                                    "3: Divorced/widowed" = "6: Divorced/widowed, not living with partner")
                              
panel_bis_sub <- select(panel_bis, pseudonym, 
                        region, region_FR, region_UK, region_DE,
                        income, income_FR_DE, income_UK,
                        education, education_DE, education_FR, education_UK,
                        family, family_UK_DE, family_FR,
                        gender, age_num)

panel_bis2 <- read_sav("data/orig/panel_bis2.sav")

panel_bis2$pseudonym <- as.integer(panel_bis2$pseudonym)
panel_bis2$hh_size <- as_factor(panel_bis2$md_1172)
panel_bis2$place_of_residence <- as_factor(panel_bis2$md_1264)
panel_bis2$home_owner <- as_factor(panel_bis2$m_1006)
panel_bis2$empl_status <- as_factor(panel_bis2$md_1181)
panel_bis2$empl_type <- as_factor(panel_bis2$md_1634)
panel_bis2$empl_job <- as_factor(panel_bis2$md_1635)

panel_bis2$empl_status2 <- fct_recode(panel_bis2$empl_status, 
                                      "Full-time" = "Work full-time (30+ hours per week)",
                                      "Part-time" = "Work part-time (up to 29 hours per week)",
                                      "In-school" = "Apprenticeship, Internship",
                                      "In-school" = "School",
                                      "In-school" = "Student",
                                      "In-school" = "Re-training",
                                      "Unemployed" = "Currently unemployed",
                                      "Not-working" = "Pensioner/retired, formerly in full-time work",
                                      "Not-working" = "Not working (housewife/house husband)",
                                      "Not-working" = "Maternity leave, Parental leave, Sabbatical")

panel_bis2$empl_status2[panel_bis2$empl_status2 == "- please select -"] <- NA

panel_bis2$home_owner[panel_bis2$home_owner == "0"] <- NA

panel_bis2 <- droplevels(panel_bis2)

panel_bis2_sub <- select(panel_bis2, pseudonym, hh_size, place_of_residence, home_owner, empl_status, empl_status2, empl_type, empl_job)

# Merge

df.socdem <-
  panel_sub %>%
  left_join(panel_bis_sub, by = "pseudonym") %>%
  right_join(panel_bis2_sub, by = "pseudonym") # keep cases with complete background data

# Fill in missing info

df.socdem$age <- ifelse(is.na(df.socdem$age), df.socdem$age_num, df.socdem$age)
df.socdem$gender <- as.factor(df.socdem$gender.x)
df.socdem$gender <- as.factor(ifelse(is.na(df.socdem$gender), as.character(df.socdem$gender.y), as.character(df.socdem$gender)))
df.socdem$gender.y <- NULL
df.socdem$gender.x <- NULL

saveRDS(df.socdem, "./data/work/sociodemo.rds")
