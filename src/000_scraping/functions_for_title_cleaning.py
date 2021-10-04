import re

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

def split_title(title) : 
    if  title != None and '|' in title :
        t = title.split('|')
        t = [i for i in t if len(i) == max([len(x) for x in t])]
        return t[0]
    if title and ' - ' in title : 
        t = title.split(' - ')
        t = [i for i in t if len(i) == max([len(x) for x in t])]
        return t[0]
    
    return title 

def filter3(x) : 
    drop_char = ['picture of the day','video report','in pictures','podcast','video profile','video','as it happened','STEPHEN POLLARD','EXPRESS COMMENT','World News','The Independent' ,'Daily Record' ,'Cornwall Live' ,'AOL','BBC', 'Using the BBC','Barking and Dagenham Post', 'Daily Star', 'Birmingham Live', 'Mirror Online', 'CBBC Newsround']#  , 'BBC - ', ' - Using the BBC    '
    for i  in drop_char:
        if re.search(rf"(\[|-|–) ?{i}((\w+)$|.?|\])", str(x), re.IGNORECASE):
            p = re.sub(rf"(\[|-|–) ?{i}((\w+)$|.?|\])", '', str(x).lower())
            return p
    return x

def title_filter(title) : 
    filtred_title = clean(title)
    filt = split_title(filtred_title).strip()
    filt3 = clean(filter3(filt))

    return filt3
