# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:38:47 2019

@author: mezaour
"""

#####

from bert_serving.client import BertClient
import datetime as dt
import numpy as np 
import pandas as pd 



#######
def get_vect(df) :
    print('\n Bert process... \n')
    print('Check that the Bert server is launched! \n')
    ######################### bert encoding  
    port = 6006
    port_out = 6007
    bc = BertClient(port=port, port_out=port_out, show_server_config=True)
    ##Check that the Bert server is launched!
    #bc = BertClient(ip='18.184.240.39', check_version=False)  
    lnew = []
    debut = dt.datetime.now()
    nb = 1
    print('\n------------------------------------------------------------------\n')
    for i in df['filtred_title'].values :
        if i != None and i != '' : 
            lnew.append(bc.encode([i]))
            nb = nb+1
        
        if (nb)%10==0 : 
            temps = dt.datetime.now() - debut
            reste = len(df['filtred_title'].values)-nb
            tmp_rest = (reste*temps)/nb
            print("\n [",nb,"/",len(df['filtred_title'].values),"] -- temps -- ", temps, " --  reste -- ",reste ," -- temps restant --", tmp_rest)
        
    vec = np.vstack(lnew)
    vec = pd.DataFrame(vec)
    
    
    
    bc.close()
    
    return vec
