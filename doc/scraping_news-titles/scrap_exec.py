# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:27:31 2019

@author: Denis
"""

urls_df = pd.read_csv('exemple_url.csv',sep=";")

## scrap_media will scrap contents for urls in urls_df['url']
## 'title_only': scraps just the title or tries to scrap more
## the task is done by packs of size 'pack_size'
## after each pack the result is incrementally saved in file_name

scrap_media(urls_df, file_name = 'scrap_titles_exemple1.csv', pack_size = 10, title_only = True)
scrap_media(urls_df, file_name = 'scrap_all_exemple2.csv', pack_size = 10, title_only = False)


