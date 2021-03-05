# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:41:20 2019

@author: mezaour
"""




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

