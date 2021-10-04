# from sklearn.externals import joblib #### sklearn 0.21.1 version "pip install scikit-learn==0.21.1"
import joblib
import re

#################
def set_df(url) : 
    #######
    ### five predicitive features
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

def url_filter_app(dflink,tree,rf) : 
    check_url = set_df(dflink)
    c = ['url', 'domain', 'nchar', 'number_occr', 'slash_occ', 'slash_ratio', 'dash_occr', 'dash_ratio']
    check_url = check_url[c]
    ff = check_url.iloc[: , 2:8].values
    tree_home = tree.predict(ff)
    rf_home   = rf.predict(ff)
    
    check_url['tree_home'] = tree_home
    check_url['rf_home'] = rf_home
    check_url['dump_url'] = check_url['tree_home'].values * check_url['rf_home'] 
    
    return check_url


def url_filter(dflink) : 
    check_url = dflink.drop_duplicates(subset = ['url'])
    check_url.reset_index(drop = True, inplace = True) 
    ### load the model from disk
    tree = joblib.load('data/finalized_dtree_model.sav')
    rf = joblib.load('data/finalized_rforest_model.sav')
    
    tree_uk = joblib.load('data/finalized_dtree_model_uk.sav')
    rf_uk = joblib.load('data/finalized_rforest_model_uk.sav')
    subset_df = check_url.loc[check_url.country == 'UK',]
    if not subset_df.empty:
        res = url_filter_app(subset_df[['url', 'domain']],tree_uk, rf_uk)
        check_url.loc[check_url.country == 'UK', 'dump_url'] =  res['dump_url']
    
    subset_df = check_url.loc[check_url.country != 'UK',]
    if not subset_df.empty:
        res = url_filter_app(subset_df[['url', 'domain']],tree,rf)
        check_url.loc[check_url.country != 'UK', 'dump_url'] = res['dump_url']
    
    drop_ind = [x for x in check_url['dump_url'].index if check_url['dump_url'].iloc[x] == 1]
    check_url.drop(drop_ind, inplace = True)
    check_url.reset_index(drop = True, inplace = True)
          
    full = check_url[['domain', 'country', 'url']]
    
    
    return full, check_url
