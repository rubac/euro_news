#!/usr/bin/env python
# coding: utf-8

# In[11]:


# imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[12]:


# load France data
fr_df = pd.read_csv("FR.csv", sep=';')
fr_df = fr_df.dropna()
fr_df.head()


# In[13]:


# LDA using French title
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_df=0.8, min_df=2)
document_term_mat_french = vectorizer.fit_transform(fr_df['title_fr'].values.astype('U'))
# apply LDA
from sklearn.decomposition import LatentDirichletAllocation
# try 5 components
LDA = LatentDirichletAllocation(n_components=5, random_state=15)
LDA.fit(document_term_mat_french)
# add topics to document term matrix
topic_values_french = LDA.transform(document_term_mat_french)
fr_df['topic_french'] = topic_values_french.argmax(axis=1)
fr_df.head()


# In[14]:


print('Using LDA with count vectorizer and the French titles,')
for i,topic in enumerate(LDA.components_):
    print(f'10 most likely words for topic {i}:')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[15]:


# LDA using English titles
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
document_term_mat_english = vectorizer.fit_transform(fr_df['title_en'].values.astype('U'))
# apply LDA
from sklearn.decomposition import LatentDirichletAllocation
# try 5 components
LDA = LatentDirichletAllocation(n_components=5, random_state=15)
LDA.fit(document_term_mat_english)
# add topics to document term matrix
topic_values_english = LDA.transform(document_term_mat_english)
fr_df['topic_english'] = topic_values_english.argmax(axis=1)
fr_df.head()


# In[16]:


print('Using LDA with count vectorizer and the English titles,')
for i,topic in enumerate(LDA.components_):
    print(f'10 most likely words for topic {i}:')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[34]:


# export LDA topics
fr_df.to_csv(r'fr_lda_topics.csv')


# In[20]:


# import data again
fr_df2 = pd.read_csv("FR.csv", sep=';')
fr_df2 = fr_df2.dropna()


# In[21]:


# NMF with French titles (combined with Tfidf)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2)
document_term_mat_french = tfidf_vect.fit_transform(fr_df2['title_fr'].values.astype('U'))

from sklearn.decomposition import NMF
nmf = NMF(n_components=5, random_state=15)
nmf.fit(document_term_mat_french)

# concat topics
topic_nmf_french = nmf.transform(document_term_mat_french)
fr_df2['topic_french'] = topic_nmf_french.argmax(axis=1)
fr_df2.head()


# In[22]:


print('Using NMF with TFIDF vectorizer and the French titles,')
for i,topic in enumerate(nmf.components_):
    print(f'10 most likely words for topic {i}:')
    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[24]:


# NMF with English titles (combined with Tfidf)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
document_term_mat_english = tfidf_vect.fit_transform(fr_df2['title_en'].values.astype('U'))

from sklearn.decomposition import NMF
nmf = NMF(n_components=5, random_state=15)
nmf.fit(document_term_mat_english)

# concat topics
topic_nmf_english = nmf.transform(document_term_mat_english)
fr_df2['topic_english'] = topic_nmf_english.argmax(axis=1)
fr_df2.head()


# In[25]:


print('Using NMF with TFIDF vectorizer and the English titles,')
for i,topic in enumerate(nmf.components_):
    print(f'10 most likely words for topic {i}:')
    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[39]:


# export NMF topics
fr_df2.to_csv(r'fr_nmf_topics.csv')

