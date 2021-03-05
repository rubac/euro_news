# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:45:57 2019

@author: mezaour
"""

from bs4 import BeautifulSoup
import datetime as dt
import pandas as pd 
import re
from progress.bar import Bar
import urllib.request as request
import csv
### ########
#####clean tags !!!!
def clean(tag) :
    import re 
    tag = str(tag)
    if tag != 'None' : 
        tag = re.sub(r"^\W", '', tag)
        tag = tag.replace('/', ';') 
        tag = tag.replace('\n','')
        tag = tag.replace('  ',' ')
        tag = tag.strip()
        tag = '; '.join([x.strip() for x in tag.split(';')])
        return tag
    return None
#############
def get_Title(soup):
    if soup.title : 
        return soup.title.string
    
    return None 
#############
def get_Tags(soup):
    #Scrap du TAG 
    tg = ''
    tags = None
    meta = soup.find_all('meta')
    val_bool = True
    
    for tag in meta:
        if 'name' in tag.attrs.keys() and tag.attrs['name'].strip().lower() in ['keywords', 'news_keywords']:
            if val_bool == True and tags == None: 
                try : 
                    tg =  tg + '/' + tag.attrs['content']
                    val_bool = False
                except KeyError : 
                    tg  = None
            tags = tg
        
        if 'property' in tag.attrs.keys() and tag.attrs['property'].strip().lower() in ['article:tag'] : 
            tg = ''
            tg =tg +'/'+ tag.attrs['content']
            tags = tg
    
    
    if soup.find('div', class_ = 'article-tags mnid-bnews_art_tags plid-0') : 
        tg = soup.find('div', class_ = 'article-tags mnid-bnews_art_tags plid-0').find_all('li', class_ = 'lnid-')
        tags = ''
        for t in tg : 
            tags = tags + '/' + t.text
    
    if soup.find('div', class_ = 'LongArticle-info') :
        try : 
            tg = soup.find('div', class_ = 'LongArticle-info').find_all('div')
            tags = ''
            for t in tg :
                tags = t.text + tg
            
        except : 
            tags = None
    
    if  soup.find('div', class_ = 'sp-c-sport-navigation sp-c-sport-navigation--secondary qa-secondary') :
            try : 
                tags = soup.find('div', class_ = 'sp-c-sport-navigation sp-c-sport-navigation--secondary qa-secondary').text
            except : 
                tags = None
    if soup.find('ul', class_ = "tags-list") : 
        try : 
            for t in soup.find('ul', class_ = "tags-list") : 
                tg = t.text + '/'+ tg
            tags = tg
        except : 
            tags = None
    if tags != None: 
        return clean(tags)
    
    return None
#############
def get_Date(soup):    
    #Scrap de la date
    date = None
    meta = soup.find_all('meta')                
    for m in meta :
        if 'property' in m.attrs.keys() and m.attrs['property'].strip().lower() in ['article:published_time','publish-date', 'og:pubdate', 'rnews:datepublished', 'datepublished'] :
                #print("La date : \n",m.attrs['content'])
                date = m.attrs['content']
        elif 'name' in m.attrs.keys() and m.attrs['name'].strip().lower() in ['dcsext.articlefirstpublished','article:published_time','publish-date', 'dcterms.created'] :
                #print("La date : \n",m.attrs['content'])
                date = m.attrs['content']
    
    if date == None :
        if soup.find('p', class_ = 'updated') :             
            date_format=re.search("(0[1-9]|1[0-9]|2[0-9]|3[0-1])(.|-)([a-zA-Z]+)(.|-|)20[0-9][0-9]",soup.find('p', class_ = 'updated').text)
            if date_format.group() : 
                date = dt.datetime.strptime(str(date_format.group()), '%d %B %Y')
        else : 
             date = None
    #get Date in div tag
    div_tags = ["date date--v2", 'ideas-video-information__publish-date']
    for t in div_tags : 
        if soup.find('div', class_ =  t) : 
                try : 
                    date = soup.find('div', class_ = t).text
                    date_format3 = re.search("(0?[1-9]|1[0-9]|2[0-9]|3[0-1])(.|-)(0[1-9]|1[0-2]|[a-zA-Z]+)(.|-|)(20[0-9][0-9])", date)
                    
                    if date_format3.group() : 
                        date = dt.datetime.strptime(str(date_format3.group()), '%d %B %Y')
                except : 
                    date = None
    #get Date in span tag              
    if soup.find('span', class_ = 'article-meta__timestamp-date') : 
        date = soup.find('span', class_ = 'article-meta__timestamp-date').text
    #get Date in time tag
    if soup.find('time', class_ = 'Dateline') : 
        date = soup.find('time', class_ = 'Dateline').text   
    return date
#############
def get_Resume(soup):    
    #Scrap du resume 
    resume = None
    meta = soup.find_all('meta')
    for m in meta :
        if 'name' in m.attrs.keys() and m.attrs['name'].strip().lower() in ['description'] :
            try : 
                resume = m.attrs['content']
            
            except KeyError : 
                resume = None
    
    if soup.find('p', class_ = 'LongArticle-synopsis gel-double-pica') : 
        try : 
            
            if soup.find('p', class_ = 'LongArticle-synopsis gel-double-pica').text : 
                resume = soup.find('p', class_ = 'LongArticle-synopsis gel-double-pica').text
        except : 
            resume = None
    
    if resume != None and "\n" in resume : 
        resume = str(resume).replace('\n', ' ')
        
    resume = clean(resume)
    return resume 
#############
def get_Text (soup):
    texte = None
    list_tag = ['Article-content ','Article-content has-dropCap','text--prose','text','l-article__container__container','component-content','recipe-description','content','promo','post_content','articleBody','article-body',
                'article-body-content standard-body-content','article-content','content-a','entry-content',
                'story-content  p402_premium clearfix','p402_premium', 'story-body sp-story-body gel-body-copy', 'field-body', 'story-body__inner','Article-content ']
    
    for tag in list_tag : 
        if soup.find_all('div', itemprop = tag): 
            body = soup.find_all('div', itemprop = tag)
            texte = ''
            for  b  in body : 
                paragraphes = b.find_all('p')
                for p in paragraphes :
                    texte = texte +' '+ p.text
                
        elif soup.find_all('div', class_ = tag) :
            body = soup.find_all('div',  class_ = tag)
            texte = ''
            for b in body :
                paragraphes = b.find_all('p')
                for p in paragraphes : 
                    texte = texte +' '+ p.text
            #cas ou y'a absence de balise <p>
                if texte == '' and b.text : 
                    texte = b.text
    tg_section = ['Article-body Article-container Theme--business', 'Article-body Article-container Theme--news','Article-body Article-container',
                  'Article-body Article-container Theme--ireland', 'Article-body Article-container Theme--world',
                  'Article-body Article-container Theme--sport', 'Article-body Article-container Theme--money', 'Article-body Article-container Theme--times2',
                  'Article-body Article-container Theme--','Article-body Article-container Theme--travel']             
    for tg in tg_section : 
            if soup.find('section', class_ = tg) : 
                try : 
                    body = soup.find('section', class_ = tg).findChildren('p')
                    iter_p = iter(body)
                    next(iter_p)
                    t = ''
                    for p in iter_p:
                        t = p.text+' '+t
                    texte = t
                except :
                    texte = None
                    
    texte = clean(texte)
    return texte
#############
def get_Type(soup) :     
    type_art = '' 
    meta = soup.find_all('meta')
     
    for m in meta : 
        if 'name' in m.attrs.keys() and m.attrs['name'].strip().lower() in ['tmgads.channel','page type','channel','article:section','category'] : 
            type_art_m = m.attrs['content']
            type_art = type_art + '/' + type_art_m 
        elif 'property' in m.attrs.keys() and m.attrs['property'].strip().lower() in ['og:site_name','og:section','article:section'] : 
            if type_art != '' : 
                type_art = type_art+'/'+m.attrs['content']
            else : 
                type_art = m.attrs['content']
    
    if soup.find('ul', class_ = 'article-tags') :
        type_art = soup.find('ul', class_ = 'article-tags')
        type_art = str(type_art.text)
        
    
    if soup.find_all('a') : 
        les_types = soup.find_all('a')
        for t in les_types:
            if 'class' in t.attrs.keys() and [x for x in ['channel-link', 'active-channel','page-title__logo'] if [x] in t.attrs.values()] != [] :
                #print(t.attrs.get('title'))
                type_art = str(t.attrs.get('title'))
                
               
    if soup.find_all('a') : 
        for t in soup.find_all('a') :
            if 'class' in t.attrs.keys() and [x for x in [['active', 'item'], ['channel-link'], ['active-channel'], ['page-title__logo']] if x in t.attrs.values() ]!= [] : 
                if t.attrs.get('title') != None : 
                    type_art = str(t.attrs.get('title'))
                else : 
                    type_art = t.text +'/'+ type_art
    
    #get type in li tag
    lst_class = ['active', 'stick']
    for l in lst_class : 
        if soup.find('li', class_ = l) and type_art == None :
            type_art = ''
            if soup.find('li', class_ = l) : 
                try :    
                    t1 = soup.find('li', class_ = l).find('a').text
                    type_art = type_art + str(t1)
                except : 
                    t1 = soup.find('li', class_ = l).find('a')
                    type_art = type_art + str(t1)
                if soup.find('div', class_ = 'navbar-secondary-container') : 
                    try: 
                        t2 = soup.find('div', class_ = 'navbar-secondary-container').find('li', class_ = 'active').find('a').text
                        type_art = type_art + ' '+ str(t2)
                    except: 
                        type_art = str(t1)
    
    if soup.find('a', id = 'brand') and type_art != None: 
        try : 
            type_art = soup.find('a', id = 'brand').text
        except : 
            type_art = None
            
    type_art = clean(type_art)       
    return type_art

#%%

def run_scrap (list_links, title_only) : 
    
    data_list = []
    n = 0
    
    with Bar('Processing', max=len(list_links)) as bar:
            # Do some work
            for l in list_links['url'] :  
                try : 
                    '''
                    link ="https://www."+l 
                    data_dic = {}
                    r = requests.get(link)
                    soup = BeautifulSoup(r.content, "html")
                    '''
                    # if the website have a bot detect 
                    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
                    #headers = {}
                    data_dic = {}
                    # page you're about to scrape!
                    link = "https://www."+l
                    # open the page
                    page_request = request.Request(link, headers=headers)
                    page = request.urlopen(page_request)
                    
                    # parse the html using beautifulsoup
                    html_content = BeautifulSoup(page, 'html.parser')
                    soup = html_content
                    
                    data_dic['Title'] = get_Title(soup)
                    data_dic['links'] = link
                    if not  title_only : 
                        data_dic['Tags']  = get_Tags(soup)
                        data_dic['Date']  = get_Date(soup)
                        data_dic['Resume']= get_Resume(soup)
                        data_dic['Text']  = get_Text(soup)
                        data_dic['Type']  = get_Type(soup)
                    
                    data_list.append(data_dic)
                        
                    bar.next()
                    
                except : 
                    n = n + 1
                    continue
                    
            print('\n Failures:  --------->',n)
            return data_list
        
# ###### writing result function
#file_name set a file name, 'scrap_Titles.csv' default file name 
#data_dic : data result as a dictionary
#title_only : scraping methode, if title_only = True only titles will be scraped
            
def write_csv (file_name, data_dic, title_only) : 
        import os
        with open(file_name, 'a', encoding='utf-8') as csvFile:
            
            fields = ['Date','Title', 'Resume', 'Text', 'Type', 'Tags', 'links']
            if title_only :
                fields = ['Title', 'links']
            writer = csv.DictWriter(csvFile, fieldnames=fields, delimiter = ';', lineterminator = '\n')
            try : 
                if os.stat(file_name).st_size == 0 : 
                    writer.writeheader()
                #csv.Sniffer().has_header(open("scraped_links.csv").read())
                for data in data_dic:
                    writer.writerow(data)
                
            except  FileNotFoundError as e  : 
                print('------------ chech the file path ----------', e)
        
        csvFile.close()
        print('Writing Success !! :) ')
    
# ###### function that launch the scraping process
#list_links : as ['url'] column datafram 
#file_name set a file name, 'scrap_Titles.csv' default file name 
#pack_size : the size of one package you want, 1000 as default size
#title_only : scraping methode, if title_only = True only titles will be scraped 

def scrap_media(list_links, file_name = 'scrap_Titles.csv', pack_size = 1000, title_only = False) : 
   
    num_pack = 0
    debut = dt.datetime.now()
    nbr_pack = len([ i for i in range(pack_size, len(list_links), pack_size)])
    nbr = 0
    for i in range(pack_size, len(list_links)+1, pack_size) : 
        if (len(list_links)-i >= pack_size) :
            num_pack = num_pack + 1
            print("--------------------------------------- Scraping pack Num° : ", num_pack , "  --------------------------------------------")
            pack = run_scrap(list_links[i-pack_size : i], title_only)
            write_csv(file_name, pack, title_only)
            
            nbr+=1
            temps = dt.datetime.now() - debut
            print('  temps ecoule:  --  ', temps ,'  -- temps restant : -- ',((nbr_pack + 1 + -nbr)*temps/nbr), " -- reste :  -- ",(nbr_pack +1- nbr))
            
        else :
            num_pack = num_pack + 1
            
            print("--------------------------------------- Scraping pack Num° : ", num_pack , "  --------------------------------------------")
            pack = run_scrap(list_links[i-pack_size : len(list_links)], title_only)
            write_csv(file_name, pack, title_only)
            
            nbr+=1
            temps = dt.datetime.now() - debut
            print('  temps ecoule:  --  ', temps ,'  -- temps restant : -- ',((nbr_pack + 1-nbr)*temps/nbr), " -- reste :  -- ",(nbr_pack + 1-nbr))
            
#%%
    

    