# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:44:24 2018

@author: XuL
"""

import re
import pandas as pd
from nltk import tokenize




def find_case(x):    
    
    try:
        st = re.search(r'Bankr\.|Bank\.',  x).span()[0]
        end = re.search(r'Case No + \d\d_\d\d\d\d\d|\d_\d\d\d\d\d',  x).span()[1]
        case = x[st:end]
    except:   
        try:
            rge = re.search(r'Case No +\d\d_\d\d\d\d\d|\d_\d\d\d\d\d',  x).span()
            st = rge[0]
            end = rge[1]
            case = x[st:end]
        except:
             try:
                rge = re.search(r' and +\d\d_\d\d\d\d\d|\d_\d\d\d\d\d',  x).span()
                st = rge[0]
                end = rge[1]
                case = x[st:end]
             except:
                case = ""  
    x = x.replace(case,"")  
    
    return x
    
    
    
    
    
def remov_case(x):
    
    new_x = find_case(x)
    while new_x != x:
        x = new_x
        new_x = find_case(x)
        
    return new_x
    
    
    
    
def pre_process(text):
    
    # remove web address
    try:
        st = re.search(r'http',  text).span()[0]
        end = re.search(r'\.+(net|com)',  text).span()[1]
        s = text.replace(text[st:end],"")
    except:
        s = text   
    
    # remove dashed line in title
    search = re.search(r'--------', s)
    if not (pd.isnull(search)):
        st = search.span()[0]
        ed = st
        while s[ed] == '-':
            ed += 1
        s = re.sub(s[(st-1):(ed)],'.',s)
    
    
    # substitude hyphen in joint words
    s = re.sub(r'--',',',s)
    s = re.sub(r'-','_',s)
    
    # remove backslash
    s = re.sub(r'/','',s)
    
    # remove comma before and dot after 
    s = re.sub(r', Inc\.', ' INC', s)
    s = re.sub(r' Inc\.', ' INC', s)
    s = re.sub(r' INC,', ' INC', s)
    s = re.sub(r', INC', ' INC', s)
    s = re.sub(r'Incs', 'INC\'s', s)
    
    s = re.sub(r', Esq\.', ' Esq', s)
    s = re.sub(r' Esq\.', ' Esq', s)
    s = re.sub(r' Esq,', ' Esq', s)
    s = re.sub(r', Esq', ' Esq', s)
    
    s = re.sub(r', L\.L\.C\.', ' LLC', s)
    s = re.sub(r' L\.L\.C\.', ' LLC', s)
    s = re.sub(r' LLC\.', ' LLC', s)
    s = re.sub(r' LLC,', ' LLC', s)
    s = re.sub(r', LLC', ' LLC', s)
    
    s = re.sub(r', L\.P\.', ' LP', s) 
    s = re.sub(r' L\.P\.', ' LP', s) 
    s = re.sub(r' LP\.', ' LP', s)  
    s = re.sub(r' LP,', ' LP', s)
    s = re.sub(r', LP', ' LP', s)
    
    s = re.sub(r', P\.C\.',' PC', s)
    s = re.sub(r' P\.C\.',' PC', s)
    s = re.sub(r' PC\.',' PC', s)
    s = re.sub(r' PC,',' PC', s)
    s = re.sub(r', PC',' PC', s)
    
    s = re.sub(r', P\.A\.',' PA', s)
    s = re.sub(r' P\.A\.',' PA', s)
    s = re.sub(r' PA\.',' PA', s)
    s = re.sub(r' PA,',' PA', s)
    s = re.sub(r', PA',' PA', s)
        
    s = re.sub(r'General Partnership', 'GP', s)
    s = re.sub(r', GP', ' GP', s)
    s = re.sub(r' GP,', ' GP', s)        
    
    s = re.sub(r', APC', ' APC', s)
    s = re.sub(r' APC,', ' APC', s)
    
    s = re.sub(r' No\.', ' No', s)
    s = re.sub(r' Nos\.', ' No', s)
    s = re.sub(r' Nos', ' No', s)
    
    s = re.sub(r' et.\ al\.', ' et al', s)
    s = re.sub(r' et al\.', ' et al', s)
    s = re.sub(r' et al\.', ' et al', s)
    s = re.sub(r' et al,', ' et al', s)
    s = re.sub(r', et al', ' et al', s)
    s = re.sub(r' et al', ' Et Al', s)
    
    # switch uppercase and lowercase
    s = re.sub(r' Debtors', ' debtors', s)
    s = re.sub(r' Debtor', ' debtor', s)
       
    s = re.sub(r's Motion', '\'s motion', s)
    s = re.sub(r' Motion', ' motion', s)
    
    s = re.sub(r' Is ', ' is ', s)
    s = re.sub(r' Not ', ' not ', s)
    s = re.sub(r' Cannot ', ' can not ', s)
    
    s = re.sub(r' Files', ' files', s)
    s = re.sub(r' Filed', ' filed', s)
    s = re.sub(r' File', ' file', s)
    s = re.sub(r' Filing', ' filing', s)
    s = re.sub(r', which filed ', ' filing ', s)
    
    s = re.sub(r' dba ', ' DBA ', s)
    s = re.sub(r' fdba ', ' FDBA ', s)
    s = re.sub(r' fka ', ' FKA ', s)
       
    # convert abbrivations
    s = re.sub(r' the U\.S\. Bankruptcy Court', ' the court', s)
    s = re.sub(r' the US Bankruptcy Court', ' the court', s)
    s = re.sub(r' the United States Bankruptcy Court', ' the court', s)
    s = re.sub(r' the Court', ' the court', s)
      
    
    s = re.sub(r' Corp\.', ' CORP', s)
    s = re.sub(r' Co\. ', ' Co ', s)
    s = re.sub(r' Dev\.', ' Dev', s)
    s = re.sub(r' Assoc\.', ' Association', s)
    s = re.sub(r'Mil\.', 'million', s)
    s = re.sub(r' Hon\. ', ' Hon ', s)
    s = re.sub(r' Ind\. ', ' Ind ', s)
    
    
    # remove short forms
    s = s.replace("′", "'").replace("’", "'").\
    replace("won't", "will not").replace("cannot", "can not").\
    replace("can't", "can not").replace("n't", " not").\
    replace("what's", "what is").replace("'ve", " have").\
    replace("I'm", "I am").replace("'re", " are").\
    replace("%", " percent ").replace("$", " dollar ").\
    replace("'ll", " will").replace(" it's ", " its ")
     
    # remove bankruptcy case numbers
    s = remov_case(s)
    
    # remove middle names
    s = re.sub(r'([A-Z])\.([A-Z])\.',r'\1\2',s)
    
    # remove non ASCII characters
    s = s.encode("ascii", errors='ignore')
    s = str(s, 'utf-8')
    
    # remove double commas
    s = re.sub(r" , ,", ",", s)
    
    # remove additional white spaces
    s = ' '.join(s.split())   
    
    return s
    
    
def sentence_split(text):
    return tokenize.sent_tokenize(text)





def process_filename(s):
     last = re.search("_\d\d\d\d\d\d\d\d_",s).span()[0]
     s = s[0:last]
     s = s.replace("_", " ")
     return s










