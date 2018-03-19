# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:09:16 2018

@author: XuL
"""

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import SVO 
import imp
imp.reload(SVO)
svo = SVO.SVO()

import preprocess
imp.reload(preprocess)

# Ghostscript for displaying of tree structures
import os
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
# # setup spacy parser
# =============================================================================
import en_core_web_md as spacyEn
nlp =  spacyEn.load()


# =============================================================================
# # help functions
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




def get_org(x):
   doc = nlp(x)
   return [(ent.text, ent.label_) for ent in doc.ents]




def get_ner(x):

    labels = ['ORG']
    doc = nlp(x)
    ners = [ent.text for ent in doc.ents if (ent.label_ in labels)]
    return " ".join(ners)




def get_np(x):
    
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




def get_dft_entity(x):
    
    # split sentences
    sentences = preprocess.sentence_split(x)
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
   
    ner.sort()
    
    final = []
    for i in range(len(ner)): 
        count = 0
        for j in range(i+1,len(ner)): 
            if ner[i] in ner[j]:
                count += 1
        if count == 0:
            final.append(ner[i])
    
    final = " ".join(final)
    return final




#def get_dft_entity_spcy(x):
#    
#    # split sentences
#    sentences = preprocess.sentence_split(x)
#    # find relavance
#    relavance = [svo.match_keywords(sent,keywords,keyverbs) for sent in sentences]
#
#    ner =[]
#    # if there are relavant sentences
#    if sum(relavance)>0:
#        ner = [get_ner(sentences[j]) for j in range(len(relavance)) if relavance[j]==1]
#             
#        # if spacy ner not working for relavant sentences, use spacy ner for irelavant sentences
#        if sum([1 if svo.if_rm(n) else 0 for n in ner]) == len(ner):
#            ner = [get_ner(sentences[j]) for j in range(len(relavance)) if relavance[j]==0]
#                 
#        # svo working for relavant sentences
#        else:
#            ner = [n for n in ner if not svo.if_rm(n)]
#            
#    # if all sentences are irelavant
#    else:
#        ner = [get_ner(sent) for sent in sentences ]
#   
#    ner.sort()
#    
#    final = []
#    for i in range(len(ner)): 
#        count = 0
#        for j in range(i+1,len(ner)): 
#            if ner[i] in ner[j]:
#                count += 1
#        if count == 0:
#            final.append(ner[i])
#    
#    final = " ".join(final)
#    return final

    
# =============================================================================
# extract entity from one sentence
# =============================================================================



sent = "Canejas, S.E., a single asset real estate, filed a Chapter 11 bankruptcy petition Bankr. D. P.R. Case No. 16-02644 on April 4, 2016."

sent = preprocess.pre_process(sent)
print(get_dft_entity(sent))



# =============================================================================
# # extract entities from a list of sentences
# =============================================================================

import pandas as pd


data = pd.read_excel("C:\\Users\\XuL\\Desktop\\Default_Entity_recognition\\extraction_output_v2.0.xlsx")
data = data.loc[data["keyword"].isin(keywords)]
# pre process sentences
data['prc_section'] = data['section'].apply(lambda x: preprocess.pre_process(x))


data.loc[(data['named_entity_internal']==' ')|(pd.isnull(data['named_entity_internal'])), 'named_entity_internal'] = ''
data['named_entity_internal'] = data['named_entity_internal'].apply(lambda x: preprocess.pre_process(x))

data.loc[(data['named_entity_spacy']==' ')|(pd.isnull(data['named_entity_spacy'])), 'named_entity_spacy'] = ''
data['named_entity_spacy'] = data['named_entity_spacy'].apply(lambda x: preprocess.pre_process(x))

data.loc[(data['filename']==' ')|(pd.isnull(data['filename'])), 'filename'] = ''
data['filename'] = data['filename'].apply(lambda x: x if x == '' else preprocess.process_filename(x))     

# remove duplicated rows
data = data[['named_entity_internal', 'named_entity_spacy','filename','section', 'prc_section']]
data.drop_duplicates(inplace=True)


data['ih2'] = data['prc_section'].apply(lambda x: get_dft_entity(x))
#data['spacy'] = data['prc_section'].apply(lambda x: get_dft_entity_spcy(x))

data = data[['filename','ih2','named_entity_internal','named_entity_spacy','spacy','section']]
data['acc_ih2'] = data.apply(lambda x: 100 if x[0]==x[1] else svo.accuracy_score(x[0].lower(), x[1].lower()), axis=1)
data['acc_ih'] = data.apply(lambda x: 100 if x[0]==x[2] else svo.accuracy_score(x[0].lower(), x[2].lower()), axis=1)
data['acc_spacy'] = data.apply(lambda x: 100 if x[0]==x[3] else svo.accuracy_score(x[0].lower(), x[3].lower()),  axis=1) 
#data['acc_spacy2'] = data.apply(lambda x: 100 if x[0]==x[4] else svo.accuracy_score(x[0].lower(), x[4].lower()),  axis=1) 

    
    
data.to_csv("result.csv",index=False)
# =============================================================================
# # shutdown jvm
# =============================================================================
#b.shutdown()
   


