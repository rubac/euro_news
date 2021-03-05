library(tidyverse)
setwd("/home/r_uma_2019/respondi_eu/")

URL1 <- readRDS("./data/work/pc_url.RDS")

# Need to make sure we only have logs before election
URL1 <- URL1 %>% 
  filter(used_at<="2019-05-26 00:00:01")

# extract search terms from variable 'url'
URL1$search <- str_match(URL1$url, "(?:&|\\?)q=(.*?)&")[,2]
URL1$search2 <- str_match(URL1$url, "(?:&|\\?)url=(.*?)&")[,2]
URL1$search3 <- str_match(URL1$url, "(?:&|\\#)q=(.*?)&")[,2]
URL1$url.2 <- paste0(URL1$url, '¥')
URL1$search4 <- str_match(URL1$url.2, "(?:&|\\#|\\?)q=(.*?)¥")[,2]
# some special cases have different regular expressions to capture search terms 
# (e.g. use Yen at end of url to extract search terms that are at the very end

# set empty values to NA
URL1$search[URL1$search==""] <- NA
URL1$search2[URL1$search2==""] <- NA
URL1$search3[URL1$search3==""] <- NA
URL1$search4[URL1$search4==""] <- NA

# if first (broadest, most general) approach does not capture the search term, 
# overwrite with other, more specialised versions.
URL1$search[is.na(URL1$search)] <- URL1$search2[is.na(URL1$search)]
URL1$search[is.na(URL1$search)] <- URL1$search3[is.na(URL1$search)]
URL1$search[is.na(URL1$search)] <- URL1$search4[is.na(URL1$search)]

# Clean search terms.
URL1$search <- str_to_lower(URL1$search)
URL1$search <- str_replace_all(URL1$search, "\\+", " ")
URL1$search <- str_replace_all(URL1$search, "%20", " ")
URL1$search <- str_replace_all(URL1$search, "%c3%9f", "ß")
URL1$search <- str_replace_all(URL1$search, "%c3%b6|%c3%96", "ö")
URL1$search <- str_replace_all(URL1$search, "%c3%bc|%c3%9c", "ü")
URL1$search <- str_replace_all(URL1$search, "%c3%a4|%c3%84", "ä")
URL1$search <- str_replace_all(URL1$search, "%c3%a9|%c3%89", "é")
URL1$search <- str_replace_all(URL1$search, "%c3%a8|%C3%88", "è")
URL1$search <- str_replace_all(URL1$search, "%c3%ab|%c3%8b", "ë")
URL1$search <- str_replace_all(URL1$search, "%c3%aa|%c3%8a", "ê")
URL1$search <- str_replace_all(URL1$search, "%c3%a7|%c3%87", "ç")
URL1$search <- str_replace_all(URL1$search, "%c3%a6|%c3%86", "æ")
URL1$search <- str_replace_all(URL1$search, "%c3%a1", "á")
URL1$search <- str_replace_all(URL1$search, "%c3%a2|%c3%82", "â")
URL1$search <- str_replace_all(URL1$search, "%c3%a0|%c3%80", "à")
URL1$search <- str_replace_all(URL1$search, "%c3%ae|%c3%8e", "î")
URL1$search <- str_replace_all(URL1$search, "%c3%af|%c3%8f", "ï")
URL1$search <- str_replace_all(URL1$search, "%c5%93|%c5%92", "œ")
URL1$search <- str_replace_all(URL1$search, "%c3%b4|%c3%94", "ô")
URL1$search <- str_replace_all(URL1$search, "%c3%b9|%c3%99", "ù")
URL1$search <- str_replace_all(URL1$search, "%c3%bb|%c3%9b", "û")
URL1$search <- str_replace_all(URL1$search, "%c3%be|%c5%b8", "ÿ")
URL1$search <- str_replace_all(URL1$search, "%27", "'")
URL1$search <- str_replace_all(URL1$search, "%2c", ",")
URL1$search <- str_replace_all(URL1$search, "%3f", "?") # question
URL1$search <- str_replace_all(URL1$search, "%c2%a7", "§") # legal
URL1$search <- str_replace_all(URL1$search, "%24", "$") # dollar
URL1$search <- str_replace_all(URL1$search, "%2f", "/") # slash
URL1$search <- str_replace_all(URL1$search, "%26", "&") # and
URL1$search <- str_replace_all(URL1$search, "%2b", "+") # plus
URL1$search <- str_replace_all(URL1$search, "%3d", "=") # equal
URL1$search <- str_replace_all(URL1$search, "%40", "@") # ad
URL1$search <- str_replace_all(URL1$search, "%22", '"') # quote
URL1$search <- str_replace_all(URL1$search, "%3a", ':') # :
URL1$search <- str_replace_all(URL1$search, "%09", ' ') # tab
URL1$search <- str_replace_all(URL1$search, "%23", '#') # #  
URL1$search <- str_replace_all(URL1$search, "%0a", '') # line feed  
URL1$search <- str_replace_all(URL1$search, "%5b", '[') # bracket open
URL1$search <- str_replace_all(URL1$search, "%5d", ']') # bracket close  
URL1$search <- str_replace_all(URL1$search, "%5e", '^') # ^
URL1$search <- str_replace_all(URL1$search, "%5c", '\\') # \
URL1$search <- str_replace_all(URL1$search, "%7b", '{') # {
URL1$search <- str_replace_all(URL1$search, "%7c", '|') # |
URL1$search <- str_replace_all(URL1$search, "%7d", '}') # }
URL1$search <- str_replace_all(URL1$search, "%7e", '~') # tilde
URL1$search <- str_replace_all(URL1$search, "%3e", '>') # greater than
URL1$search <- str_replace_all(URL1$search, "%b4", "'") # acute accent
URL1$search <- str_replace_all(URL1$search, "%60", "`") # acute accent
URL1$search <- str_replace_all(URL1$search, "%25", '%') # percent


# flag search engines
# dummy variable for search engines in dataset (based on most common ones)
URL1$searchengine <- 0
URL1$searchengine[str_count(URL1$url, 
                            "google.de|google.com|bing.|asknow.|ecosia.|suche.gmx|
                                 suche.web|qwant.|search.avira.|nortonsafe.search.|wow.com|
                                 izito.|wolframalpha.|duckduckgo.|suche.aol.|baidu.|
                                 search.yahoo.|dogpile.|yippi.|ask.com|webcrawler.com|
                            search.com|ixquick.com|excite.com|info.com")>0]<- 1


# exclude entries that do not have anything to do with searching
URL1$searchengine[str_count(URL1$url, 
                            c("translate.google|googleads|maps.google|
                                   google.de/maps|google.com/maps/|google.com/earth|
                                   mail.google|play.google|keep.google|accounts.google|
                                   admin.google|analytics.google|drive.google|plus.google|
                                   support.google|news.google|chrome.google|calendar.google.|
                                   myaccount.google|docs.google|adwords.google|madeby.google|
                                   passwords.google|myactivity.google|productforums.google|
                                   sites.google|google.com/calendar|newsstand.google.|
                                   photos.google.|payments.google.|inbox.google.|
                                   business.google.|aboutme.google.|accounts.google.|
                                   google.com/gmail|withgoogle.com|developers.google.|
                                   researchbing.|google.com/chrome|script.google.|
                                   -seo-google.|google.de/adwords|hangouts.google.|
                                   contacts.google.|adssettings.google.|google.com/ads|
                                   groups.google.com|get.google.|source=bing|source=google"))>0]<- 0

URL1 <- URL1 %>% 
  select(-c(search2, search3, search4, url.2))


# separate cases that come from websearch page.
URL_small.searches <- URL1[URL1$searchengine == 1,c("panelist_id","url","used_at","active_seconds",
                                                         "search","country")]
URL_small.searches <- URL_small.searches[!(is.na(URL_small.searches$search)),]


saveRDS(URL_small.searches, file = "./data/work/dat_searchterms.RDS")

#### 

URL_small.searches <- readRDS(file = "./data/work/dat_searchterms.RDS")

URL_small.searches <- URL_small.searches[(which(nchar(URL_small.searches$search) > 1)),]





