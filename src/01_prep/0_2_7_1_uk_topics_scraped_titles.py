#!/usr/bin/env python
# coding: utf-8

# In[44]:


# imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[45]:


# load UK data
uk_df = pd.read_csv("UK.csv", sep=';')
uk_df = uk_df.dropna()
uk_df.head()


# In[46]:


# LDA using filtered title
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
document_term_mat_filtered = vectorizer.fit_transform(uk_df['title_filtered'].values.astype('U'))
# apply LDA
from sklearn.decomposition import LatentDirichletAllocation
# try 5 components
LDA = LatentDirichletAllocation(n_components=5, random_state=15)
LDA.fit(document_term_mat_filtered)
# add topics to document term matrix
topic_values_filtered = LDA.transform(document_term_mat_filtered)
uk_df['topic_filtered'] = topic_values_filtered.argmax(axis=1)
uk_df.head()


# In[31]:


print('Using LDA with count vectorizer and the filtered titles,')
for i,topic in enumerate(LDA.components_):
    print(f'10 most likely words for topic {i}:')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[32]:


# LDA using regular title
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
document_term_mat_regular = vectorizer.fit_transform(uk_df['title'].values.astype('U'))
# apply LDA
from sklearn.decomposition import LatentDirichletAllocation
# try 5 components
LDA = LatentDirichletAllocation(n_components=5, random_state=15)
LDA.fit(document_term_mat_regular)
# add topics to document term matrix
topic_values_regular = LDA.transform(document_term_mat_regular)
uk_df['topic_regular'] = topic_values_regular.argmax(axis=1)
uk_df.head()


# In[33]:


print('Using LDA with count vectorizer and the regular titles,')
for i,topic in enumerate(LDA.components_):
    print(f'10 most likely words for topic {i}:')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[34]:


# export LDA topics
uk_df.to_csv(r'uk_lda_topics.csv')


# In[ ]:


# import data again
uk_df2 = pd.read_csv("UK.csv", sep=';')
uk_df2 = uk_df2.dropna()


# In[35]:


# NMF with filtered titles (combined with Tfidf)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
document_term_mat_filtered = tfidf_vect.fit_transform(uk_df2['title_filtered'].values.astype('U'))

from sklearn.decomposition import NMF
nmf = NMF(n_components=5, random_state=15)
nmf.fit(document_term_mat_filtered)

# concat topics
topic_nmf_filtered = nmf.transform(document_term_mat_filtered)
uk_df2['topic_filtered'] = topic_nmf_filtered.argmax(axis=1)
uk_df2.head()


# In[36]:


print('Using NMF with TFIDF vectorizer and the filtered titles,')
for i,topic in enumerate(nmf.components_):
    print(f'10 most likely words for topic {i}:')
    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[37]:


# NMF with regular titles (combined with Tfidf)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
document_term_mat_regular = tfidf_vect.fit_transform(uk_df2['title'].values.astype('U'))

from sklearn.decomposition import NMF
nmf = NMF(n_components=5, random_state=15)
nmf.fit(document_term_mat_regular)

# concat topics
topic_nmf_regular = nmf.transform(document_term_mat_regular)
uk_df2['topic_regular'] = topic_nmf_regular.argmax(axis=1)
uk_df2.head()


# In[38]:


print('Using NMF with TFIDF vectorizer and the regular titles,')
for i,topic in enumerate(nmf.components_):
    print(f'10 most likely words for topic {i}:')
    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[39]:


# export NMF topics
uk_df2.to_csv(r'uk_nmf_topics.csv')

