require(stringr)
source(file = "train_data_for_BERT_ft_aux.R")

########## BERT training as tag classifier
########## constitution of train, dev and test sets on the basis of scraping
#####################################################



## load article titles and tags obtained thru scraping
## title_tags is based on scrapping of top news website in the UK from september to december 2018 visiting by the full UK panel
## I used this instead of the tags for our sample articles to have more data for the training
#########
#########

data_tt = read.csv2("title_tags.csv",encoding="WINDOWS-1252",stringsAsFactors = FALSE)



## creation of a dictionnary of tags
#########

## minimal cleaning of tags
data_tt$Tags = str_replace(data_tt$Tags ,"[ ]{2,}"," ")
data_tt$Tags = tolower(data_tt$Tags)


## ordering tags by frequency
batches =  split(data_tt$Tags,sample(x=c(1:10),size=nrow(data_tt),replace=TRUE))
tags = lapply(batches,str_split,pattern="; ")
tags = do.call(c,lapply(tags,unlist))
tags = trimws(tags)


dt = tt.to.df(table(tags))
names(dt)=c("tags","freq")
dt$tags = as.character(dt$tags)

## more thorough cleaning of tags
spurious.tags = tolower(c("","nan","ctp_video","headlines","BBC","i Player","TV","tv","uktv","Fixtures Tables Video Averages Engl",
            "autoplay_video","Crosswords","News","news","tvshowbiz","femail",
            "autoplay video ctp video autoplay video ctp video","instagram"))
spurious.tags2 = c("edition","subscribe","media player","golden globes","golden globe","telegraph puzzles",
              "content of this","scottish sun","terms of service","affiliate links","autoplay_video",
              "ctp_video","blowjob",
              "support team","bbc","copyright","external sites","current content","latest news","crossword","free trial")
dt = dt[which(!nchar(dt$tags)<2&!dt$tags%in%spurious.tags),]
dt = dt[which(!str_detect(dt$tags,paste(spurious.tags2,collapse="|"))),]

## keep only tags with at least nn hits
nn = 10
dt = dt[-which(dt$freq<nn),]

## We get 1394 different tags with at least 21 articles
print(paste("Number of tags after cleaning",nrow(dt)))



## create a tag classification data frame ("echantillon") for BERT training
#########

data_tt$index = 10001:(10000+nrow(data_tt))

## data frame to be filled for each tag
echantillon0 = data.frame(matrix(NA,nrow=nn*3,ncol=5)) 
names(echantillon0)=c("Quality","#1 ID","#2 ID","#1 String","#2 String")

## for each tag, get a data frame with nn positive cases and 2*nn negative cases
## positive cases are cases when the tag was used for the article, so they are always true positives
## negative cases are cases when the tag was not used, but it still might be relevant, so we introduce false negatives
## one could filter out manually but that noise does not seem to be harmful so I did not do any manual filtering
set.seed(1234)
echantillon = lapply(1:length(dt$tags),get_cases,tagz=dt$tags,data_tt=data_tt,echantillon0=echantillon0,nn=nn)
echantillon = do.call(rbind,echantillon)
echantillon = echantillon[which(!duplicated(echantillon)),]
rm(echantillon0)

## get rid of titles which are too short / too long so that BERT gets genuine sentences
# split along spaces and count number of splits
duv =  unlist(lapply(lapply(echantillon$`#2 String`,str_split,pattern=" "),function(ag){return(length(unlist(ag)))}))
# only keep titles with more than 4 words and fewer than 101 words
echantillon = echantillon[which(duv>4&duv<101),]

## get rid of BBC iPlayer stuff
echantillon = echantillon[which(!str_detect(echantillon$`#2 String`,"BBC iPlayer")),]

## clean strings with special characters that cause problems with BERT

echantillon$`#2 String` = str_replace_all(echantillon$`#2 String`,'\"',"")
echantillon$`#2 String` = str_replace_all(echantillon$`#2 String`,'\\\n',"")
echantillon$`#1 String` = str_replace_all(echantillon$`#1 String`,'\"',"")
echantillon$`#1 String` = str_replace_all(echantillon$`#1 String`,'\\\n',"")





## split echantillon in 3 for training (8/15), dev (2/15) and test (5/15)
#########

## split
ll = nrow(echantillon)
devc = c(1:as.integer(2/15 * ll))
trainc = c((as.integer(2/15 * ll)+1):(as.integer(10/15 * ll)))
testc  = c((as.integer(2/3 * ll)+1):ll)

dir.create("essai")

## save train and dev as tsv files
write.table(echantillon[devc,],"essai/dev.tsv",sep = "\t",fileEncoding = "utf-8",col.names = FALSE,row.names = FALSE)
write.table(echantillon[trainc,],"essai/train.tsv",sep = "\t",fileEncoding = "utf-8",col.names = FALSE,row.names = FALSE)

## prepare test file by replacing Quality column with an Index column
for.test = echantillon[testc,]
names(for.test)[1]="index"
for.test$index = 1:nrow(for.test)

## save test files ("test.tsv" for BERT and "test_with_classes" with true tags)
write.table(for.test,"essai/test.tsv",sep = "\t",fileEncoding = "utf-8",col.names = FALSE,row.names = FALSE)
write.table(echantillon[testc,],"essai/test_with_class.tsv",sep = "\t",fileEncoding = "utf-8",col.names = FALSE,row.names = FALSE)


################################
################################
############
############ instructions for fine-tuning BERT can be found here
############ https://github.com/google-research/bert#fine-tuning-with-bert
############ here is the code I used
export GLUE_DIR=/home/ubuntu/glue/denis
export BERT_BASE_DIR=/home/ubuntu/uncased_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/home/ubuntu/denis_output/cola
export OUTPUT_DIR=/home/ubuntu/denis_output/cola

cd /home/ubuntu/bert-master

python run_classifier.py   --task_name=CoLA   --do_train=true   --do_eval=true   --data_dir=$GLUE_DIR/cola  --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   --learning_rate=2e-5   --num_train_epochs=3.0   --output_dir=$OUTPUT_DIR

############ once you are done, you get the model.ckpt files to use as your fine-tuned model for title embeddings.




