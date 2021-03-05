# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:47:09 2019

@author: mezaour
"""

#%%%
import re
import pandas as pd 

def clean(txt) :
    txt = str(txt)
    if  txt != 'None' : 
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
####
#create spaces, ex : DonaldTrump => Donald Trump
def space_it(p) : 
    while re.search(r"[a-z][A-Z]|[a-z]{3,10}[A-Z][A-Z][a-z]{3,10}", p) != None  : 
        p = re.sub(re.search(r"[a-z][A-Z]|[a-z]{3,10}[A-Z][A-Z][a-z]{3,10}", p).group(0),re.search(r"[a-z][A-Z]|[a-z]{3,10}[A-Z][A-Z][a-z]{3,10}", p).group(0).replace('', ' ').strip(), p)
    return p 

### set sep = ';'
def tag_separator(tags) :
    if tags != None : 
        
        a = clean(re.sub(r"^\W", '', str(tags)))
        a = a.replace('/', '; ') 
        a = a.replace(',', '; ')
        a = a.replace('; ;', '; ')
        a = a.replace('  ',' ')
        a = a.replace('&',';')
        a = a.replace(' and ','; ')
        a = a.replace(';;',';')
        a = space_it(a)
        return a
    return tags


################
###drop duplicated tags
def drop_dublic(tags_f) :
    tags_f = tags_f.replace(',', '; ')
    d = tags_f.split(';')
    d = '; '.join(list(set(d))).strip()

    return d
################

def _filter_tag(x) : 
    drop_car = ['STEPHEN POLLARD','EXPRESS COMMENT','World News','The Independent' ,'Daily Record' ,'Cornwall Live' ,'AOL','BBC', 'Using the BBC','Barking and Dagenham Post', 'Daily Star', 'Birmingham Live', 'Mirror Online', 'CBBC Newsround']
    car2 = ['Bournemouth Echo', 'Breitbart', 'Latest Barking and Dagenham News', 'Barking and Dagenham Post', 'Daily Mail Online', 'The Independent', 'HuffPost', 'The Guardian']
    car3 = ['BBC', 'CBBC', 'bbc', 'cbbc']
    
    for i  in drop_car:
        if  re.search(r" ?"+i+"( (\w+)$| ?)", str(x)): 
            return re.sub(r" ?"+i+"( (\w+)$| ?)", '', str(x))
    for j in car2 : 
        if  re.search(r" ?; "+j+"",str(x)) :
            return re.sub(r" ?\| "+j+"", '', str(x))
    for k in car3 : 
        if re.search(r"^"+k+"( \w*| | \w* \w*| \w*\d*);", str(x)) : 
            return re.sub(r"^"+k+"( \w*| | \w* \w*| \w*\d*);", '', str(x))  
    return x
############# 
### run the tags filter
def tag_filter(tags) : 
    tags_f = pd.Series([tag_separator(x) for x in tags])
    tags_f2 = [_filter_tag(x) for x in tags_f]
    tags_f2 =  [clean(re.sub(r"^\W", '', str(x))) for x in tags_f2]
    tags_f2 = pd.Series([drop_dublic(x) for x in tags_f2])
    
    return tags_f2

#############################
