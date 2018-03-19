# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:10:21 2018

@author: XuL
"""

from datetime import datetime
from itertools import chain
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import os

import SVO 
import imp
imp.reload(SVO)
svo = SVO.SVO()

import preprocess

# Ghostscript for displaying of tree structures
path_to_gs = "C:\\Program Files\\gs\\gs9.22\\bin"
os.environ['PATH'] += os.pathsep + path_to_gs



# =============================================================================
# #setup berkeley parser
# =============================================================================

import berkeleyinterface as b
# Allow entering a number for kbest parses to show when running
kbest = 1

# This should be the path to the Berkeley Parser jar file
cp = r'C:\\Users\\XuL\\stanford\\berkeley_ner\\BerkeleyParser-1.7.jar'

# Always start the JVM first!
b.startup(cp)

# Set input arguments
# See the BerkeleyParser documentation for information on arguments
gr = r'C:\\Users\\XuL\\stanford\\berkeley_ner\\eng_sm6.gr'
args = {"gr":gr, "tokenize":True, "kbest":kbest,"binarize":True,"keepFunctionLabels":False,'maxLength':999999}

# Convert args from a dict to the appropriate Java class
opts = b.getOpts(b.dictToArgs(args))

# Load the grammar file and initialize the parser with our options
bparser = b.loadGrammar(opts)



# =============================================================================
# # setup stanford parser
# =============================================================================

# Stanford Parser
os.environ['STANFORD_PARSER'] = 'C:\\Users\\XuL\\stanford\\stanford-parser-full-2017-06-09\\stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'C:\\Users\\XuL\\stanford\\stanford-parser-full-2017-06-09\\stanford-parser-3.8.0-models.jar'
from nltk.parse import stanford


# Java path
java_path = "C:/Program Files/Java/jre1.8.0_152/bin"
os.environ['JAVAHOME'] = java_path


# Stanford NER
_model_filename = 'C:\\Users\\XuL\\AppData\\Local\\Continuum\\anaconda3\\stanford_ner\\classifiers\\english.all.3class.distsim.crf.ser.gz'
_path_to_jar = 'C:\\Users\\XuL\\AppData\\Local\\Continuum\\anaconda3\\stanford_ner\\stanford-ner.jar'
from nltk.tag.stanford import StanfordNERTagger
st = StanfordNERTagger(model_filename=_model_filename, path_to_jar=_path_to_jar)
sparser = stanford.StanfordParser()



# =============================================================================
# # setup spacy parser
# =============================================================================
import en_core_web_md as spacyEn
nlp =  spacyEn.load()


# =============================================================================
# load data
# =============================================================================


keywords = ["chapter_11_bankruptcy",
"filed_chapter_11",
"chapter_11_petition",
"chapter_11_reorganization",
"sought_bankruptcy_protection",
"defaulted_on"]


keyverbs = ['file','filed','filing','files',\
            'seek','seeks','sought','seeking',\
            'default','defaulted','defaulting','defaults',\
            'estimate','estimates','estimating','estimated',\
            'commence','commences','commencing','commenced']



data = pd.read_excel("C:\\Users\\XuL\\Desktop\\NLP\\extraction_output_v2.0.xlsx")


data = data.loc[data["keyword"].isin(keywords)]


# =============================================================================
# pre process
# =============================================================================

# pre process sentences
data['prc_section'] = data['section'].apply(lambda x: preprocess.pre_process(x))


# remove empty labels
data.loc[data['named_entity_internal']==' ', 'named_entity_internal'] = np.NaN
data.loc[data['named_entity_spacy']==' ', 'named_entity_spacy'] = np.NaN


# remove incorrect labels
rm_lst = ['No entity found', 'each',  'Each of the debtors',\
          'debtor', 'The debtor', 'the debtor', 'debtors', 'The debtors', 'the debtors', \
         'Inc', 'LLC',  'It', 'it',  'She', 'she', 'He', 'he', 'They', 'they'
         'The Company', 'The company', 'the Company','the company','Company','company','']

data.loc[(data['named_entity_internal'].isin(rm_lst)),'named_entity_internal'] = np.NaN
data.loc[(data['named_entity_spacy'].isin(rm_lst)),'named_entity_spacy'] = np.NaN


# pre process previous predictions
data['named_entity_internal'] = data['named_entity_internal'].apply(lambda x: preprocess.pre_process(x) if not pd.isnull(x) else x)
data['named_entity_spacy'] = data['named_entity_spacy'].apply(lambda x: preprocess.pre_process(x) if not pd.isnull(x) else x)

     
# pre process true labels
data['filename'] = data['filename'].apply(lambda x: preprocess.process_filename(x) if not pd.isnull(x) else x)     

# remove duplicated rows
data = data[['named_entity_internal', 'named_entity_spacy','filename','section', 'prc_section']]
data.drop_duplicates(inplace=True)





# =============================================================================
# # extract NER
# =============================================================================



        
def get_org(x):
   doc = nlp(x)
   return [(ent.text, ent.label_) for ent in doc.ents]







def get_ner(x):

    labels = ['ORG']
    doc = nlp(x)
    ners = [ent.text for ent in doc.ents if (ent.label_ in labels)]
    return " ".join(ners)





def get_np (x):
    
    subject1 = get_ner(x)
    try:
        string = b.parse(bparser,opts,x).toEscapedString()
        root_tree = nltk.tree.Tree.fromstring(string)
        subject2 = svo.process_parse_tree(root_tree, \
            list(set([lemmatizer.lemmatize(ver,'v') for ver in keyverbs])) )
        if not subject2:
            subject = subject1
        else:
            tmp = []
            for s in subject2[:-1]:
                tmp.extend(s['subject'])
                tmp.extend(",")
            tmp.extend(subject2[-1]['subject'])
            subject2 = " ".join(tmp)
            subject = subject2
    except:
        subject = subject1
    return subject

    



ners = []
for i in range(len(data)):
    record = data.iloc[i]
    
    # split sentences
    sentences = preprocess.sentence_split(record['prc_section'])
    print(i,sentences)

    # find relavance
    relavance = [svo.match_keywords(sent,keywords,keyverbs) for sent in sentences]

    ner =[]
    # if there are relavant sentences
    if sum(relavance)>0:
        ner = [get_np(sentences[j]) for j in range(len(relavance)) if relavance[j]==1]

        # if svo not working for relavant sentences, using spacy ner for relavant sentences
        if sum([1 if svo.if_rm(n) else 0 for n in ner]) == len(ner):
             ner = [get_ner(sentences[j]) for j in range(len(relavance)) if relavance[j]==1]
             
             # if spacy ner not working for relavant sentences, use spacy ner for irelavant sentences
             if sum([1 if svo.if_rm(n) else 0 for n in ner]) == len(ner):
                 ner = [get_ner(sentences[j]) for j in range(len(relavance)) if relavance[j]==0]

                 
        # svo working for relavant sentences
        else:
            ner = [n for n in ner if not svo.if_rm(n)]
            
    # if all sentences are irelavant
    else:
        ner = [get_ner(sent) for sent in sentences ]
    ner = " ".join(ner)
    print(ner)
    ners.append(ner)


# consolidate results
result = pd.DataFrame({"ih2":ners, 'filename':data['filename'], "section":data['section'], \
                       "ih":data["named_entity_internal"],\
                       "spacy":data["named_entity_spacy"]})
    
 
result['acc_ih2'] = result.apply(lambda x: fuzz.token_set_ratio(x[0], x[2]), axis=1)
result['acc_ih'] = result.apply(lambda x: fuzz.token_set_ratio(x[0], x[1]) if not pd.isnull(x[1]) else 0, axis=1)
result['acc_spacy'] = result.apply(lambda x: fuzz.token_set_ratio(x[0], x[4]) if not pd.isnull(x[4]) else 0, axis=1) 

    
    
result.to_csv("result.csv",index=False)



# shutdown java virtual machine
b.shutdown()


# =============================================================================
# issues to be resolved
# =============================================================================

# # parallel entities
#    dba
#    fdba
#    and
#    aka
#    fka
#   each of
    
    

# # sentence pattern
# file application
# estimate/disclose assets/liabilities/debts
# The petition/The petitions signed manager of 
# owner of 
# remove the debtor/ the debtors


    

# # how to remove -----------------
    # done


# how to parse bankr. Case No.


# check if LLC/INC/GP/PC has been tagged as NP
# not working if the NE does not end with LLC
# check if spacy's NE is part of Stanford's ?
#    Done




# a good way to identify correctness



























