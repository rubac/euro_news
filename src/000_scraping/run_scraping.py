##### import packages 
import pandas as pd 
import os
import time
import datetime

##### import custom functions
import functions_for_dynamic_page_filtering as uf
import functions_for_scraping as scrap
import functions_for_title_cleaning as tf

##### load toy data
data_init = pd.read_csv("data/test_url.csv", sep = ";")

##### filter out dynamic pages
filt_url, check_result = uf.url_filter(data_init)

##### scraping
urls = filt_url.url
date = datetime.datetime.now()
filename = "scrap_"+str(date).replace(':', '-')+".csv"
scrap.scrap_media_with_pool(list_links=urls, file_name= filename, pack_size=2, title_only=True)

 



