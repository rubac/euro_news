# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:39:25 2019

@author: mezaour / bonnay
"""

#### import package and functions
###################

from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser
from bert_serving.client import BertClient

##
import numpy as np
import pandas as pd 
import datetime as dt

####
import get_sentence_embeddings_aux

#### start Bert Server
####################


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


##### get BERT encoding using fine-tuned model
##### here we use the default strategy to get sentence encodings which is simply averaging over (word embeddings
######################### 

#eliminer les valeur manquantes

df = pd.read_csv('scrapUK_all.csv',sep=";")

drop_na = [x for x in df['title_filtered'].index if df['title'].iloc[x] == '']
df2 = df.drop(drop_na)
df2 =  df2.reset_index(drop = True)

bc = BertClient(port=6006, port_out=6007, show_server_config=True)

 

lnew = []

for i in df2['title'].values :

    if i != None and i != '' :

        lnew.append(bc.encode([i]))

   

vec = np.vstack(lnew)
vec = pd.DataFrame(vec)

dff = pd.concat([df,vec],axis=1)

dff.to_csv("title_mean_strategy.csv",sep=";",encoding="UTF-8",index = False)



server.close()
bc.close()
server.close() 

