# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:14:16 2019

@author: mezaour
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#################

def set_df(url) : 
    #######
    ### cinq predicitive features
    ### count: number of times the url appears in the data: already given in df
    ### slash_occ: number of slashes in the url
    ### nchar: number of characters in the url
    ### slash_ratio: slash_occ / nchar
    ### number_occ: number of numbers occuring in the url
    ###
    url['slash_occ'] = [str(x).count('/') for x in url.url]
    url['nchar'] = [len(str(x)) for x in url.url]
    url['slash_ratio'] = url['slash_occ'].values / url['nchar'].values
    url['number_occr'] = [len(re.findall(r"\d{1}", str(re.findall(r"\/{1}\S*", str(x))))) for x in url.url]
    url['dash_occr'] = [str(x).count('-') for x in url.url]
    url['dash_ratio'] = url['dash_occr'].values / url['nchar'].values
    
    
    return url
    

def url_filter(df) :
    
    df = set_df(df)
    c = [ 'url', 'nchar', 'number_occr', 'slash_occ', 'slash_ratio', 'dash_occr', 'dash_ratio', 'home']
    df = df[c]
    
    df['home'] = [True if x == "True" else False  for x in df.home]
    print('hello')

    #df['home'] = df['home'].map({True: 1, False: 0})    
    #df.home = df.home.astype(bool)
    #df['home'] = df['home'].map({True: 1, False: 0})
    
    ######  Decision Tree
    ##Train/Test 
    #### 
    tt = df.iloc[:,1:7]
    X = tt.values
    
    Y = df.values[:,7]
    Y=Y.astype('int')
    
    X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size = 0.3, random_state = 100)
    
    tree = DecisionTreeClassifier()
    tree.fit(X_train,y_train)
    pred = tree.predict(X_test)
    print("Accurary tree model on test set",metrics.accuracy_score(y_test, pred))
    
    ########################  
    ###############################################
    # save the model to disk
    #filename = 'finalized_dtree_model.sav'
    #joblib.dump(tree, filename)
    ###############################################
    ##### Random forest
    ##
    #Import Random Forest Model
    rf=RandomForestClassifier(n_estimators=50)
    #Train the model 
    rf.fit(X_train,y_train)
    pred = rf.predict(X_test)
    #accuracy
    print("Accurary RF model on test set",metrics.accuracy_score(y_test, pred ))

    ###############################################
    # save the model to disk
    #filename = 'finalized_rforest_model.sav'
    #joblib.dump(rf, filename)
    
    return tree,rf


def url_filter_app(dflink,tree,rf) : 
    check_url = set_df(dflink)
    c = ['url', 'domain', 'nchar', 'number_occr', 'slash_occ', 'slash_ratio', 'dash_occr', 'dash_ratio']
    check_url = check_url[c]
    ###prediction 
    ff = check_url.iloc[: , 2:8].values
    tree_home = tree.predict(ff)
    rf_home   = rf.predict(ff)
    
    check_url['tree_home'] = tree_home
    check_url['rf_home'] = rf_home
    check_url['dump_url'] = check_url['tree_home'].values * check_url['rf_home'] 
    #check_url = check_url.loc[check_url.dump_url==0]
    
    
    return check_url
#%%


    


    