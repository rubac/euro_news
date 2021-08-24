library(stringr)
library(qdapDictionaries)
source(file = "train_data_for_BERT_ft_aux.R")


########## BERT training as tag classifier
########## constitution of training data
########## input: data frame with titles and tags
########## output: train, dev and test sets for BERT fine-tuning
#####################################################



## load data
load("data_title.RData")
data_tt = data_title
names(data_tt)[which(names(data_tt)=="tags_en")]="Tags"
names(data_tt)[which(names(data_tt)=="title_en")]="Title"

## clean tags
data_tt$Tags = str_replace_all(data_tt$Tags,"/","; ")
data_tt$Tags = str_replace_all(data_tt$Tags,"\\+"," ")
data_tt$Tags = str_replace_all(data_tt$Tags,",","; ")
data_tt$Tags = str_replace_all(data_tt$Tags,"; $","")
data_tt$Tags = str_replace_all(data_tt$Tags,"& fixturestables","")
data_tt$Tags = str_replace_all(data_tt$Tags,"& fixturestable","")
data_tt$Tags = str_replace_all(data_tt$Tags,"facts.divers","")
data_tt$Tags = str_replace_all(data_tt$Tags,"-","")




data_tt$Tags = str_replace_all(data_tt$Tags,"&amp|(at bild)|scoresresultscalendarorder of play|scores gossiptransfersall teamsleagues & cupsfa cupwomeneuropean|2\\.","")
data_tt = data_tt[which(!data_tt$Tags==""),]
data_tt = data_tt[which(!data_tt$Tags=="nan"),]
data_tt$Tags = str_replace_all(data_tt$Tags,"^; ","")
data_tt$Tags = str_replace_all(data_tt$Tags ,"[ ]{2,}"," ")
data_tt$Tags = tolower(data_tt$Tags)

## creation of a dictionnary of tags ordered by frequency
batches =  split(data_tt$Tags,sample(x=c(1:10),size=nrow(data_tt),replace=TRUE))
tags = lapply(batches,str_split,pattern="; ")
tags = do.call(c,lapply(tags,unlist))
tags = trimws(tags)
head(tags)
dt = tt.to.df(table(tags))
names(dt)=c("tags","freq")
dt$tags = as.character(dt$tags)

## delete meaningless tags and keep 10.000 most common tags
spurious.tags = tolower(c("","nan","ctp_video","headlines","BBC","i Player","TV","tv","uktv","Fixtures Tables Video Averages Engl",
            "autoplay_video","crosswords","news","news","tvshowbiz","femail","??? - ???","+++","9-1-1 (1",
            "autoplay video ctp video autoplay video ctp video","instagram","video","fb-instantarticles","huffpost","les aravis","currently","news","headlines",
            "others","dsl","t-dsl","desk-delta","boursorama","digit.wdr.de"))
spurious.tags2 = c("edition","subscribe","media player","golden globes","golden globe","telegraph puzzles",
              "content of this","scottish sun","terms of service","affiliate links","autoplay_video",
              "ctp_video","blowjob","dailymail","iplayer","itv",
              "support team","bbc","copyright","external sites","current content","latest news","crossword","free trial")
dt = dt[which(!nchar(dt$tags)<2&!dt$tags%in%spurious.tags),]
dt = dt[which(!str_detect(dt$tags,paste(spurious.tags2,collapse="|"))),]
dt = dt[1:10000,]

## keep only tags which occur at least nn times in the data
nn= 35
dt = dt[which(dt$freq>=nn),]
print(paste("Number of tags with at least",nn,"occurrences in the data:",nrow(dt)))



## create a tag classification data frame ("echantillon") for BERT training
## for each tag, get a data frame with nn positive cases and 2*nn negative cases
## positive cases are cases when the tag was used for the article, so they are always true positives
## negative cases are cases when the tag was not used, but it still might be relevant, so we introduce false negatives

data_tt$index = 10001:(10000+nrow(data_tt))
echantillon0 = data.frame(matrix(NA,nrow=nn*3,ncol=5))
names(echantillon0)=c("Quality","#1 ID","#2 ID","#1 String","#2 String")

set.seed(1234)
zz <<- 0
echantillon = lapply(1:length(dt$tags),get_cases,tagz=dt$tags,data_tt=data_tt,echantillon0=echantillon0,nn=nn)
echantillon = do.call(rbind,echantillon)
echantillon = echantillon[which(!duplicated(echantillon)),]
ecs = echantillon
#View(echantillon)
head(echantillon)
rm(echantillon0)

prop.table(table(echantillon$Quality))

## get rid of titles which are too short / too long so that BERT gets genuine sentences
## only keep titles with more than 4 words and fewer than 101 words
duv =  unlist(lapply(lapply(echantillon$`#2 String`,str_split,pattern=" "),function(ag){return(length(unlist(ag)))}))
summary(duv>4&duv<30)
echantillon = echantillon[which(duv>4&duv<30),]

## clean strings with special characters that cause problems with BERT
echantillon$`#2 String` = str_replace_all(echantillon$`#2 String`,'\"',"")
echantillon$`#2 String` = str_replace_all(echantillon$`#2 String`,'\\\n',"")
echantillon$`#1 String` = str_replace_all(echantillon$`#1 String`,'\"',"")
echantillon$`#1 String` = str_replace_all(echantillon$`#1 String`,'\\\n',"")
echantillon$`#1 String` = trimws(echantillon$`#1 String`)
echantillon$`#2 String` = trimws(echantillon$`#2 String`)

nrow(echantillon)


## split echantillon in 3 for training (6/10), dev (2/10) and test (2/10)
#########

## split

lr = sample(c(1:nrow(echantillon)),nrow(echantillon),replace=FALSE)
trainc = lr[c(1:as.integer(6/10 * length(lr)))]
devc = lr[c((as.integer(6/10 * length(lr))+1):(as.integer(8/10 * length(lr))))]
testc  = lr[c((as.integer(8/10 * length(lr))+1):length(lr))]

dir.create("tags_feb_21")

## save train and dev as tsv files
write.table(echantillon[devc,],"tags_feb_21/dev_121.tsv",sep = "\t",fileEncoding = "utf-8",col.names = FALSE,row.names = FALSE)
write.table(echantillon[trainc,],"tags_feb_21/train_121.tsv",sep = "\t",fileEncoding = "utf-8",col.names = FALSE,row.names = FALSE)
write.table(echantillon[testc,],"tags_feb_21/test_121.tsv",sep = "\t",fileEncoding = "utf-8",col.names = FALSE,row.names = FALSE)
