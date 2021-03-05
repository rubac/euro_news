library(FactoMineR)
library(ggplot2)
library(stringr)
library(distances)
library(plyr)
library(reshape2)

source("exploratory_topic_analysis_aux.R")


###### manage title encodings
### load encodings
te = read.csv2("result_bert_UK_final/title_mean_strategy.csv",stringsAsFactors = FALSE)
#temean = read.csv2("result_bert_UK_final/title_cls_strategy.csv",stringsAsFactors = FALSE)

### delete duplicates if any
te = te[which(!duplicated(te)),]

### convert vector dimensions columns to numeric
for (i in 6:ncol(te)){te[,i]=as.numeric(te[,i])}

### load url hits
load("url_UK.RData")

### check that all encoded titles come from UK url hits by respondents
summary(te$url%in%url_UK$url)

### merge url hits with encodings
euUK = merge(url_UK[,c("url","panelist_id","used_at","domain")],te,by="url",all.x=TRUE)

### keep only one url hit by respondent
### to prevent overweighting articles manically read many many times by the same person
names(euUK)[which(names(euUK)=="panelist_id")]="pseudonym"
euUK = euUK[which(!duplicated(euUK[,c("pseudonym","url")])),]

### check results
print(paste("Successfully encoded urls:",floor(length(which(!is.na(euUK$X767)))/nrow(euUK)*100),"%"))

### filter out respondents for whom we don't have much info
### filter out articles whose titles have not been encoded
euUK$count = 1
individual.counts = aggregate(euUK$count,by=list(euUK$pseudonym),FUN=sum)
summary(individual.counts)
euUK = euUK[which(euUK$pseudonym %in% individual.counts$Group.1[which(individual.counts$x>10)]),]
euUK = na.omit(euUK)



####################
### PCA on BERT encoding
### interpreting dimensions and "who reads what"


for.PCA = euUK[,which(str_detect(names(euUK),"^X"))]
res = PCA(for.PCA)
dim = as.data.frame(res$ind$coord)
dim$index = 1:nrow(dim)
dim$pseudonym = euUK$pseudonym

### interpreting PCA dimensions
### dimension 1: football vs scary news (abuse & crime)
interpret(dim,1,euUK)
### dimension 2: TV series vs Brexit
interpret(dim,2,euUK)
### dimension 3: Europen Elections vs (sex) crime
interpret(dim,3,euUK)
### dimension 4: Local news vs celebrities
interpret(dim,4,euUK)
### dimension 5: sports (non football, mixed with some other international stuff) vs deals (money related, business)
interpret(dim,5,euUK)

load("survey_data.RData")
mix = merge(dim,survey_data,by="pseudonym",all.x=TRUE)
mix$v_31 = factor(mix$v_31)
levels(mix$v_31) = c(NA,NA,NA,"remain","leave")
names(mix)[which(names(mix)=="v_31")]="referendum_vote"

### variable "change" says whether they changed their mind in the last European elections (pre-election vs post-election survey)
### variable "referendum_vote" says what they said they voted (if they voted) to the 2016 EU membership referendum

### referendum x topics: leave people are more into scary news, more into brexit news, and more into sex crime as well
aggregate(mix[,paste0("Dim.",c(1:5))],by=list(mix$referendum_vote),FUN=mean)

# making a picture
# each point correspond to an article, it is colored according to the behavior of the person who read it
# articles on the left are about football, they are mostly read by remain people, articles to the right are scary, they are read by leave people
# articles at the bottom are about TV series, mostly on the BBC, they are read by remain people
# articles at the top are about sex crimes, they are read by leave people
q = ggplot(transform(mix[, c("Dim.1","Dim.2")], cluster = mix$referendum_vote), 
           aes(x = Dim.1, y = Dim.2, colour = cluster)) +
  geom_point() + scale_colour_manual(values=c("purple", "orange"))

q

### change in voting intention: people who changed are much less into brexit news, and they are also more into celebs than local news
### ---> people who change their mind are less interested in both national and local politics
aggregate(mix[,paste0("Dim.",c(1:5))],by=list(mix$change),FUN=mean)

# making a picture
# each point correspond to an article, it is colored according to the behavior of the person who read it
# articles at the bottom and to the right are politics related (Brexit / local news), they are mostly read by people who don't change their mind
q = ggplot(transform(mix[, c("Dim.2","Dim.4")], cluster = mix$change), 
           aes(x = Dim.2, y = Dim.4, colour = cluster)) +
           geom_point() + scale_colour_manual(values=c("purple", "orange","gray100","gray100"))

q


##########################
############# VERY MESSY STUFF JUST FOR ME TO BE ASHAMED OF
############# CLUSTERING ARTICLES

################# determination du nombre optimal de cluster
encodings = euUK[,which(str_detect(names(euUK),"^X"))]


wss <- 2:10
for (i in 2:10) wss[i-1] <- sum(kmeans(encodings,
                                       centers=i)$withinss)
plot(2:10, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

################# creation des clusters dim 1 et 2 après normalisation

nbclust <- 9
kmeans.comment <- kmeans(encodings,nbclust)
cluster.kmeans <- as.factor(kmeans.comment$cluster)
table(cluster.kmeans)
centers = kmeans.comment$centers
nrow(centers)

tyti = find.typical.titles(encodings,centers,z=50,euUK)
View(tyti)

ac = data.frame(clusters = cluster.kmeans,pseudonym = euUK$pseudonym)
levels(ac$clusters) =c("general news","sports","Brexit","TV series","money","celebs","crime","football","entertainment")
ac$count = 1
ac2 = aggregate(ac$count,by=list(ac$pseudonym,ac$clusters),FUN=sum,drop=FALSE)
names(ac2) = c("pseudonym","cluster","count")
head(ac2)



# Specify id.vars: the variables to keep but not split apart on
ac2w <- dcast(ac2, pseudonym ~ cluster, value.var="count")
ac2w$total = rowSums(ac2w[,2:ncol(ac2w)])
for (i in 2:(ncol(ac2w)-1)){
  ac2w[,i] = ac2w[,i] / ac2w$total
}

mix2 = merge(ac2w,survey_data,by="pseudonym",all.x=TRUE)
mix2$v_31 = factor(mix2$v_31)
levels(mix2$v_31) = c(NA,NA,NA,"remain","leave")
names(mix2)[which(names(mix2)=="v_31")]="referendum_vote"
mix2$change = factor(mix2$change)
levels(mix2$change) = c("changed vote","did not change",NA,NA)

mix2s = na.omit(mix2[,c("pseudonym","referendum_vote",as.character(levels(ac$clusters)))])
mix2l = melt(mix2s, id.vars=c("pseudonym", "referendum_vote"))
head(mix2l)


df2 <- data_summary(mix2l, varname="value", 
                    groupnames=c("variable", "referendum_vote"))
df2$referendum_vote=as.factor(df2$referendum_vote)


p<- ggplot(df2, aes(x=variable, y=value, fill=referendum_vote)) + 
  geom_bar(stat="identity", color="black", 
           position=position_dodge())
p

##########
mix3s = na.omit(mix2[,c("pseudonym","change",as.character(levels(ac$clusters)))])
mix3l = melt(mix3s, id.vars=c("pseudonym", "change"))

df2 <- data_summary(mix3l, varname="value", 
                    groupnames=c("variable", "change"))
df2$change =as.factor(df2$change)


p<- ggplot(df2, aes(x=variable, y=value, fill=change)) + 
  geom_bar(stat="identity", color="black", 
           position=position_dodge())
print(p)