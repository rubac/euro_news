# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:01:32 2019

@author: bonnay / mezaour
"""

#%%
## import  packages

import pandas as pd
import numpy as np
import datetime

import scrap_script_V4 as scrap
import titres_filter as tf
import filter_v1 as filt
import bert_ft
import tags_filter as tg_filt

from mtranslate import translate

from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser
from bert_serving.client import BertClient


#########################
#########################
##### URL MANAGEMENT
#%%
## create the 'dynamic content' detection models on the basis of the training data tbf in checking_learn
## (we actually combine two models tree and rf)

dftrain = pd.read_csv('checking_learn2.csv', sep = ';', usecols = ['url', 'home'])
tree,rf = filt.url_filter(dftrain)

#%%
## load urls and filter out dynamic content pages
## load
data = pd.read_csv('uDE2.csv', sep = ';')
print("Number of unique urls before filtering: " + str(len(set(data.url.values))))
####call the url_filter function which uses the tree and rf model to filter out home pages
data = filt.url_filter_app(data,tree,rf)
data  = data.loc[data.dump_url==0]
print("Number of unique urls after filtering: " + str(len(set(data.url.values))))

#########################
#########################
##### SCRAPING
#%%
## scraping urls
## the result is written as filename after each batch
## title_only = True will only scrap titles



date = datetime.datetime.now()
filename = "scrap_fri1.csv"
filename = "scrap_fri2.csv"

title_only=False
scrap.scrap_media(data,filename, pack_size=50, title_only=title_only)


#%%
df = pd.read_csv(filename, sep = ';', header=0,encoding = 'unicode_escape')
df.head()
df.columns

try :
    df = df.dropna(subset = ['Title']).reset_index(drop = True)
    ## call the title_filter
    tit_filt = tf.title_filter(df.Title)

    if df.shape[0] == tit_filt.shape[0] :
        df['filtred_title'] = tit_filt
        if not title_only :
            tags = df.Tags
            tags_filt = tg_filt.tag_filter(tags)
            df['filtred_tags'] = tags_filt
    else :
        print('title_filter error ! check the size of title_filter results')

except :
    print ('title_filter_error!')


#
df = df[df['links'].notnull()]
df = df[df['filtred_title'].notnull()]
df['url'] = [i[12:] for i in df.links.values] 

df = pd.merge(data, df, on = 'url', how = 'inner')

print(str(len(df))+" out of "+str(len(data))+" titles scraped")

#########################
#########################
##### TRANSLATION
#%%
### translate titles from German to English

titres = list(df.filtred_title.values)
trans = []
i = 0
while i < len(titres):
    nt = translate(titres[i],"en","de")
    trans.append(nt)
    i = i+1
    if (i%20==0):
        print(str(i)+"/"+str(len(titres)))
        print(nt)
        

df['title_en'] = trans
df.to_csv("preDE_2.csv",sep=";",encoding="UTF-8",index = False)


#%%
#filter out empty title !
a = df
drop_ind = [x for x in a['title_en'].index if a['title_en'].iloc[x] == '']
a.drop(drop_ind, inplace = True)
a.reset_index(drop = True, inplace = True )
a = a[a['title_en'].notnull()]
str(len(a))+"out of"+str(len(df))"
print(str(len(a))+" out of "+str(len(df))+" titles translated")

#########################
#########################
##### BERT ENCODING
#%%
# start BERT server

default_model_location    = 'c:/uncased_L-12_H-768_A-12'
fine_tuned_model_location = 'C:/Users/Denis/Dropbox/HCK/BERT/mrpc_mannheim'
fine_tuned_model_name     = 'model.ckpt-3545'

common = [ '-model_dir', default_model_location,
           '-tuned_model_dir', fine_tuned_model_location ,
           '-ckpt_name',fine_tuned_model_name ,
           '-cpu',
           '-num_worker', '2',
           '-port', str(6006),
           '-port_out', str(6007),
           '-max_seq_len', 'None',
           ]
      
args = get_args_parser().parse_args(common)  
server = BertServer(args)
server.start()

#%%
# data frame for BERT encoding

df2= a[['filtred_title','title_en','url']] 
df2.to_csv("preFR_2bis.csv",sep=";",encoding="UTF-8",index = False)

#%%

bc = BertClient(port=6006, port_out=6007, show_server_config=True)


lnew = []
k = 0
while k < len(df2['title_en'].values):
    i = df2['title_en'].values[k]
    if i != None and i != '' :
        lnew.append(bc.encode([i]))
    k = k + 1
    if (k%20==0):
        print(i)
        print(str(k)+"/"+str(len(titres)))

vec = np.vstack(lnew)
vec = pd.DataFrame(vec)

df2 = pd.concat([df2,vec],axis=1)

df2.to_csv("DE_2.csv",sep=";",encoding="UTF-8",index = False)
# this data frame needs to be merged with the original original URL1 data frame to recover pseudonyms of people who read the pages

bc.close()
server.close() 

#%%


#%%