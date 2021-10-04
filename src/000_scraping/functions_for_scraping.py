from bs4 import BeautifulSoup
import datetime as dt
import re
import csv
import requests
from multiprocessing.pool import ThreadPool
from functools import partial

### ########
def clean(tag) :
    if tag != 'None' or tag != 'nan' : 
        tag = tag.replace(';',',')

        return tag
    return None

#############
def get_Title(soup):
    if soup.title : 
        title = tf.title_filter(soup.title.string)
        return title
    
    return None 

#############
def get_Tags(soup):
    #Scrap TAG 
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
    if soup.find('a', class_ = 'tag_breadcrumb margin_bottom_xs margin_right_xs') : 
        try : 
            tags = soup.find('a', class_ = 'tag_breadcrumb margin_bottom_xs margin_right_xs').text
        except : 
            tags = None
    if soup.find('ul', class_ = 'breadcrumb') and tags == None:
        tg = ''
        t = soup.find('ul', class_ = 'breadcrumb').find_all('a')
        for i in t : 
            tg = i.text.strip()+'/ '+ tg
        tags = tg
    if soup.find(id="breadcrumb") and tags == None :
        try : 
            tg = soup.find(id = "breadcrumb").find('div').text
            tg = tg.replace('›', '/ ')
            tags = tg
        except : 
            tags = None 
    if soup.find('li', class_ ="ts-active-parent") and tags == None:
            b = ''
            try : 
                a = soup.find('li', class_ ="ts-active-parent").find('a').text
                if soup.find('li', class_= "ts-active") : 
                    b = soup.find('li', class_= "ts-active").text
                tags = a.strip() +'/ '+ b.strip()
            except : 
                tags = None
    if tags != None: 
        return clean(tags)
    
    return None

#############
def get_Date(soup):    
    #Scrap Date
    date = None
    meta = soup.find_all('meta')                
    for m in meta :
        if 'property' in m.attrs.keys() and m.attrs['property'].strip().lower() in ['article:published_time','publish-date', 'og:pubdate', 'rnews:datepublished', 'datepublished','og:article:published_time'] :
                date = m.attrs['content']
        elif 'name' in m.attrs.keys() and m.attrs['name'].strip().lower() in ['dcsext.articlefirstpublished','article:published_time','publish-date', 'dcterms.created'] :
                date = m.attrs['content']
    if date == None :
        if soup.find('p', class_ = 'updated') :             
            date_format=re.search("(0[1-9]|1[0-9]|2[0-9]|3[0-1])(.|-)([a-zA-Z]+)(.|-|)20[0-9][0-9]",soup.find('p', class_ = 'updated').text)
            if date_format.group() : 
                date = dt.datetime.strptime(str(date_format.group()), '%d %B %Y')
        else : 
             date = None
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
    if soup.find('span', class_ = 'article-meta__timestamp-date') : 
        date = soup.find('span', class_ = 'article-meta__timestamp-date').text
    if soup.find_all('time') : 
           try : 
                time = soup.find("time")
                date = time.attrs['datetime']
           except : 
                date = None
    if soup.find('time', class_ = 'Dateline') : 
        date = soup.find('time', class_ = 'Dateline').text   
    return date

#############
def get_Resume(soup):    
    #Scrap Resume 
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
        
    return clean(resume) 

#############
def get_Text (soup):
    texte = None
    list_tag = ['Article-content ','Article-content has-dropCap','text--prose','text','l-article__container__container','component-content','recipe-description','content','promo','post_content','articleBody','article-body',
                'article-body-content standard-body-content','article-content','content-a','entry-content',
                'story-content  p402_premium clearfix','p402_premium', 'story-body sp-story-body gel-body-copy', 'field-body', 'story-body__inner','Article-content ',
                'article__content   old__article-content-single', 'fig-content__body', 'article-section margin_bottom_article']
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
                  'Article-body Article-container Theme--','Article-body Article-container Theme--travel', 'article__content']             
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
                    
    ##############################
    if soup.find('section') and texte == None :
        texte = ''
        try : 
            txt = soup.find("section", 'article__content')
            for p in txt.find_all('p', class_ = 'article__paragraph') : 
                texte = texte +' '+ p.text
            texte = texte.replace('  ','')
        except : 
            texte = None
    return clean(texte)

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
                type_art = str(t.attrs.get('title'))
    if soup.find_all('a') : 
        for t in soup.find_all('a') :
            if 'class' in t.attrs.keys() and [x for x in [['active', 'item'], ['channel-link'], ['active-channel'], ['page-title__logo']] if x in t.attrs.values() ]!= [] : 
                if t.attrs.get('title') != None : 
                    type_art = str(t.attrs.get('title'))
                else : 
                    type_art = t.text +'/'+ type_art
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
            
    return clean(type_art)




#%%
###### writing result function
#file_name set a file name, 'scrap_Titles.csv' default file name 
#data_dic : data result as a dictionary
#title_only : scraping methode, if title_only = True only titles will be scraped
            
def write_csv (file_name, data_dic, title_only) : 
        import os
        with open(file_name, 'a', encoding='UTF-8') as csvFile:
            
            fields = ['Date','Title', 'Resume', 'Text', 'Type', 'Tags', 'url']
            if title_only :
                fields = ['Title', 'url']
            writer = csv.DictWriter(csvFile, fieldnames=fields, delimiter = ';', lineterminator = '\n')
            try : 
                if os.stat(file_name).st_size == 0 : 
                    writer.writeheader()
                #csv.Sniffer().has_header(open("scraped_links.csv").read())
                for data in data_dic:
                    if data : 
                        writer.writerow(data)
                
            except  FileNotFoundError as e  : 
                print('-- Writing...   Filed: check the file path', e)
        
        csvFile.close()
        print('  Writing...   Success !! :) ')
    

  
#%%            
# =============================================================================
#  fonction pour le multi processing
# =============================================================================

####### function that launch the scraping process
#list_links : as urls input
#file_name set a file name, 'scrap_Titles.csv' default file name 
#pack_size : the size of one package you want, 1000 as default size
#title_only : scraping methode, if title_only = True only titles will be scraped 

def scrap_media_with_pool(list_links, file_name = 'scrap_Titles.csv', pack_size = 1000, title_only = False) : 
    print('\n Scraping process...\n')
    num_pack = 0
    debut = dt.datetime.now()
    nbr_pack = len([ i for i in range(pack_size, len(list_links), pack_size)])
    nbr = 0
    for i in range(pack_size, len(list_links)+1, pack_size) : 
        if (len(list_links)-i >= pack_size) :
            num_pack = num_pack + 1
            print("\n-- Scraping... pack Num° : ", num_pack , "  --")
            ####
            ### Nombre de process que l'on souhaite avoir
            with ThreadPool(5) as pool:
                ### 'partial' redefinie la fonction run_scrap2 avec un parametres figé, ici title_only à False 
                zip_fct = partial(run_scrap2, title_only=title_only) # prod_x has only one argument x (y is fixed to 10)
                ### faire tourner la foncrion zip_fcr sur 5 process
                pack = pool.map(zip_fct, list_links[i-pack_size : i])
                ### Femer les process fini pour ne pas avoir de process fantome
            pool.terminate()
            pool.join()
            ##
            write_csv(file_name, pack, title_only)
            nbr+=1
            temps = dt.datetime.now() - debut
            print('  temps -- ', temps ,'  -- temps restant -- ',((nbr_pack + 1 + -nbr)*temps/nbr), " -- reste :  -- ",(nbr_pack +1- nbr))
            print('------------------------------------------------------------')
            
        else :
            num_pack = num_pack + 1
            
            print("\n-- Scraping... pack Num° : ", num_pack , "  --")

            with ThreadPool(5) as pool:
                zip_fct = partial(run_scrap2, title_only=title_only) # prod_x has only one argument x (y is fixed to 10)
                pack = pool.map(zip_fct, list_links[i-pack_size : len(list_links)])
            pool.terminate()
            pool.join()
            ## 
            write_csv(file_name, pack, title_only)
            nbr+=1
            temps = dt.datetime.now() - debut
            print('  temps -- ', temps ,'  -- temps restant -- ',((nbr_pack + 1 + -nbr)*temps/nbr), " -- reste :  -- ",(nbr_pack +1- nbr))
            print('------------------------------------------------------------')



def run_scrap2 (l, title_only) : 
###### run_scrap2 prend seulement un lien en argument (l), contrairement à run_scrap qui prend une liste de lien !!
    data_list = []
    n = 0
    data_dic = {}
    # page you're about to scrape!
    link = "https://www."+l
    # if the website have a bot detect 
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
    try :
        # open the page
        page_request = requests.get(link, headers=headers, timeout=5)
    except (ConnectionError, Exception) as err:
        link = "https://"+l
        try :
            page_request = requests.get(link, headers=headers, timeout=5)
        except (ConnectionError, Exception) as error:
            print( "Connection Exception is :", error)
            n = n + 1
    # make sure that the web page is available
    try : 
        if page_request.status_code == 200:
            page = page_request.text
            # parse the html using beautifulsoup
            html_content = BeautifulSoup(page, 'html.parser')
            soup = html_content
            
            data_dic['Title'] = get_Title(soup)
            data_dic['url'] = l
            if not  title_only : 
                data_dic['Tags']  = get_Tags(soup)
                #data_dic['Date']  = get_Date(soup)
                #data_dic['Resume']= get_Resume(soup)
                #data_dic['Text']  = get_Text(soup)
                #data_dic['Type']  = get_Type(soup)
            data_list.append(data_dic)
        else:
           n = n + 1 
    except Exception as e:
        print( "Exception is :", e)
        n = n + 1
            
    if len(data_dic)!=0 : 
        return data_dic
            
#%%
