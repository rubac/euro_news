#!/usr/bin/env python
# coding: utf-8


# imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# load combined data
all_df = pd.read_csv("url_bert_reduced.csv", sep=';')
all_df = all_df.dropna()
all_df.head()
all_df.shape


# filter before election data
before_election = all_df['dt'] <= '2019-05-26'
all_df = all_df[before_election]
all_df.shape


# LDA using English titles
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
document_term_mat_english = vectorizer.fit_transform(all_df['title_en'].values.astype('U'))
# apply LDA
from sklearn.decomposition import LatentDirichletAllocation
# try 20 components
LDA = LatentDirichletAllocation(n_components=20, random_state=15)
LDA.fit(document_term_mat_english)
# add topics to document term matrix
topic_values_english = LDA.transform(document_term_mat_english)
all_df['topic_english'] = topic_values_english.argmax(axis=1)
all_df.head()


print('Using LDA with count vectorizer and the English titles,')
for i,topic in enumerate(LDA.components_):
    print(f'10 most likely words for topic {i}:')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# export LDA topics
all_df.to_csv(r'all_lda_topics.csv')


# import data again
all_df2 = all_df


# NMF with English titles (combined with Tfidf)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
document_term_mat_english = tfidf_vect.fit_transform(all_df2['title_en'].values.astype('U'))

from sklearn.decomposition import NMF
nmf = NMF(n_components=20, random_state=15)
nmf.fit(document_term_mat_english)

# concat topics
topic_nmf_english = nmf.transform(document_term_mat_english)
all_df2['topic_english'] = topic_nmf_english.argmax(axis=1)
all_df2.head()


print('Using NMF with TFIDF vectorizer and the English titles,')
for i,topic in enumerate(nmf.components_):
    print(f'10 most likely words for topic {i}:')
    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# export NMF topics
all_df2.to_csv(r'all_nmf_topics.csv')

