#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


# load DE data
de_df = pd.read_csv("DE.csv", sep=';')
de_df = de_df.dropna()
de_df.head()


# In[7]:


# LDA using German titles
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_df=0.8, min_df=2)
document_term_mat_german = vectorizer.fit_transform(de_df['filtred_title'].values.astype('U'))
# apply LDA
from sklearn.decomposition import LatentDirichletAllocation
# try 5 components
LDA = LatentDirichletAllocation(n_components=5, random_state=15)
LDA.fit(document_term_mat_german)
# add topics to document term matrix
topic_values_german = LDA.transform(document_term_mat_german)
de_df['topic_german'] = topic_values_german.argmax(axis=1)
de_df.head()


# In[8]:


print('Using LDA with count vectorizer and the German titles,')
for i,topic in enumerate(LDA.components_):
    print(f'10 most likely words for topic {i}:')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[9]:


# LDA using English title
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
document_term_mat_english = vectorizer.fit_transform(de_df['title_en'].values.astype('U'))
# apply LDA
from sklearn.decomposition import LatentDirichletAllocation
# try 5 components
LDA = LatentDirichletAllocation(n_components=5, random_state=15)
LDA.fit(document_term_mat_english)
# add topics to document term matrix
topic_values_english = LDA.transform(document_term_mat_english)
de_df['topic_english'] = topic_values_english.argmax(axis=1)
de_df.head()


# In[10]:


print('Using LDA with count vectorizer and the English titles,')
for i,topic in enumerate(LDA.components_):
    print(f'10 most likely words for topic {i}:')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[34]:


# export LDA topics
de_df.to_csv(r'de_lda_topics.csv')


# In[17]:


# import data again
de_df2 = pd.read_csv("DE.csv", sep=';')
de_df2 = de_df2.dropna()


# In[18]:


# NMF with German titles (combined with Tfidf)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2)
document_term_mat_german = tfidf_vect.fit_transform(de_df2['filtred_title'].values.astype('U'))

from sklearn.decomposition import NMF
nmf = NMF(n_components=5, random_state=15)
nmf.fit(document_term_mat_german)

# concat topics
topic_nmf_german = nmf.transform(document_term_mat_german)
de_df2['topic_german'] = topic_nmf_german.argmax(axis=1)
de_df2.head()


# In[19]:


print('Using NMF with TFIDF vectorizer and the German titles,')
for i,topic in enumerate(nmf.components_):
    print(f'10 most likely words for topic {i}:')
    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[20]:


# NMF with English titles (combined with Tfidf)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
document_term_mat_english = tfidf_vect.fit_transform(de_df2['title_en'].values.astype('U'))

from sklearn.decomposition import NMF
nmf = NMF(n_components=5, random_state=15)
nmf.fit(document_term_mat_english)

# concat topics
topic_nmf_english = nmf.transform(document_term_mat_english)
de_df2['topic_english'] = topic_nmf_english.argmax(axis=1)
de_df2.head()


# In[21]:


print('Using NMF with TFIDF vectorizer and the English titles,')
for i,topic in enumerate(nmf.components_):
    print(f'10 most likely words for topic {i}:')
    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[39]:


# export NMF topics
de_df2.to_csv(r'de_nmf_topics.csv')

