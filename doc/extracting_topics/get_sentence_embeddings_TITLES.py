#!/usr/bin/python

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
def get_vect2(lst_to_encode) : 
    
    ''' Verifier que le serveur Bert est bien lancé !! '''
    #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    bv = []
    port = 6006
    port_out = 6007
    bc = BertClient(port=port, port_out=port_out, show_server_config=False)
    debut = dt.datetime.now()
    empty_vect = np.array([0]*768, dtype=np.float32).reshape(1, 768)
    print(debut)
    for i in range (0, len(lst_to_encode)) :
        print('hello')
        if i == '' :
            bv.append(empty_vect)
        else :
            bv.append(bc.encode([lst_to_encode[i]]))
       
        if (i+1)%100==0 : 
            bc.close()
            temps = dt.datetime.now() - debut
            reste = len(lst_to_encode)-i-1
            tmp_rest = (reste*temps)/i
            print("temps ecoulé --------> : ", temps, "secondes, -",i,"-  reste ----------> ",reste ," lignes, temps restant ------------> ", tmp_rest," minutes")
            bc = BertClient(port=port, port_out=port_out, show_server_config=False)
    
    #bc.close()
    return bv

#### start Bert Server
####################


default_model_location    = '/home/ubuntu/TENSOR/pretrained/'
fine_tuned_model_location = '/home/ubuntu/TENSOR/finetuned/'
fine_tuned_model_name     = 'model.ckpt-3545'

common = [ '-model_dir', default_model_location,
           '-tuned_model_dir', fine_tuned_model_location ,
           '-ckpt_name',fine_tuned_model_name ,
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

df = pd.read_csv('/home/ubuntu/TENSOR/UKDAT/scrapUK_all.csv',sep=";")

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

dff.to_csv("/home/ubuntu/TENSOR/UKDAT/title_mean_strategy.csv",sep=";",encoding="UTF-8",index = False)



server.close()
bc.close()
server.close() 

