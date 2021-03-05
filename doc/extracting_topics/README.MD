############
##### STEP 1
##### FINE-TUNING BERT
 
BERT is fine-tuned on a multiple-label classification task. The task consists in determining whether a title belongs to a certain topic.
Positive cases are obtained through scrapping articles tags (on an independent and much larger dataset of articles/tags, provided as title_tags.csv). Negative cases are obtained through negative sampling (if a tagged article was not tagged with a given tag, it is assumed that the article did not belong to that topic, this will include false negatives, but it's kinda stard in NLP in general to train like that, I believe, for example that's similar to the strategy to the negative sampling strategy for word2vec)

Train, Dev and Test sets are set-up in train_data_for_BERT.R
Instructions for fine-tuning BERT are also to be found at the end of that file

The fine-tuned model can be downloaded here
https://www.dropbox.com/sh/9g23ov5ul1kqn8n/AAC8NLTiYW-WCheDL5uff0aGa?dl=0

(training the network too about 4 hours on AWS using a p3.8xlarge machine as on https://aws.amazon.com/ec2/instance-types/p3/)

############
##### STEP 2
##### TITLE ENCODING

I used Bert As Service to quickly get sentence embeddings for each title.
Bert outputs an embedding for each word / morpheme in the title sentence and also for the sentence head marker. There are various pooling strategies to get an embedding for the sentence as a whole. I am using the default strategy which is simply averaging over the words vectors. I tried using the sentence head, results looked similar.

The python code to get the embeddings is in get_sentence_embeddings.py and the companion auxiliary file.The scrapped titles are in scrapUK_all.csv (15042 titles). Executing this gets us embeddings for each scrapped title based on the fine-tuned model which was trained during step 1.

############
##### STEP 3
##### EXPLORATORY ANALYSIS

I am simply merging the result of Step 2 (title embeddings as "title_mean_strategy.csv") with respondent information (who read which article as "url_UK.RData" x survey data about votes at the elections as "survey_data.RData"). 73% of static urls in the top 60 news websites are encoded.

I made a PCA on the title embeddings, to get interpretable dimensions. The five dimensions we get from the PCA are actually quite easy to interpret when lookins at titles which stand at the extreme on each dimension. One can they cross this with survey information about the people who read the articles positioned on the PCA. That gets the graphs I had sent during the summer.

The R code for this is in "exploratory_topic_analysis.R" with its companion auxiliary file.

A further idea would be to cluster articles, so we can say that respondent xxx read yyy articles about bla where bla is one of the clusters, or that zzz percent of the articles read by respondent xxx belong to cluster bla. I have included some code for that part at the end of the aforementioned file but it's *very* messy, I am not cleaning it because I am not very happy with the results yet. (the results are not statistically significant, SDs are huge)








