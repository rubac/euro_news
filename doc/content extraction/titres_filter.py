# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:08:37 2019

@author: mezaour
"""
import pandas as pd 
import re 

############
def clean(txt) :
    txt = str(txt)
    if txt != 'None' : 
        txt = txt.strip()
        txt = txt.replace('\t',' ')
        txt = txt.replace('\n','')
        txt = txt.replace('\n\n','')
        txt = txt.replace('\a',' ')
        txt = txt.replace('\b','')
        txt = txt.replace('  ',' ')
        txt = txt.replace(';', ',')
        txt = txt.replace('–', '-')
        txt = txt.replace('▷', '')
        
        
        return txt
    return None

def split_title(titre) : 
    if  titre != None and '|' in titre :
        t = titre.split('|')
        t = [i for i in t if len(i) == max([len(x) for x in t])]
        titre = t[0]
    if titre and ' - ' in titre : 
        t = titre.split(' - ')
        t = [i for i in t if len(i) == max([len(x) for x in t])]

        return t[0]
    elif titre and ' -' in titre : 
        t = titre.split(' -')
        t = [i for i in t if len(i) == max([len(x) for x in t])]
       
        return t[0]
    
    return titre 

def filter3(x) : 
    drop_car = ['AUDIO VIDEO FOTO BILD','Bild.de','-Handy','podcast','video profile','video','as it happened','STEPHEN POLLARD','EXPRESS COMMENT','World News','The Independent' ,'Daily Record' ,'Cornwall Live' ,'AOL','BBC', 'Using the BBC','Barking and Dagenham Post', 'Daily Star', 'Birmingham Live', 'Mirror Online', 'CBBC Newsround']#  , 'BBC - ', ' - Using the BBC    '
    
    for i  in drop_car:
        if  re.search(r"-|– "+i+"((\w+)$|.?)", str(x)):
            p = re.sub(r"-|– "+i+"((\w+)$|.?)", '-', str(x))
            return p
    return x

### run title filter 
def title_filter(titre) : 
    filtred_title = pd.Series([clean(x) for x in titre])
    filt = pd.Series([split_title(x).strip() for x in filtred_title])
    filt3 = pd.Series([clean(filter3(x)) for x in filt])

    return filt3







