# -*- coding: utf-8 -*-
"""
This file covers all the classes and functions for performing 
Named Entity Recognition on text objects. 
We developed an in-house NER tool based primarily on POS tags. 
We compare our results against state-of-art NLP tools designed and 
calibrated by Stanford University NLP Group. 

There are several dependencies to this file and we will list down 
each of them below:
    - NLTK
    - Java/Java SDK
    - Standard POS Tagger 
        https://nlp.stanford.edu/software/tagger.shtml#Download
    - Stanford NER Tagger 
        https://nlp.stanford.edu/software/CRF-NER.shtml#Download
    - Stanford Neural-Network Based Dependency Parser 
        https://nlp.stanford.edu/software/lex-parser.html
    - sklearn
    
Instructions to install each of these packages will be given later 

The rationale for looking into dependency parsing is to see if 
there is a way to identity entities in the parsed sentences.
Initial analysis shows promise, but it's premature to conclude anything 
based on this. 

@author: SankisaR

"""

import os 
#from docx import Document
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from nltk.parse import stanford
from nltk.parse.stanford import StanfordDependencyParser
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
import spacy 
from spacy.symbols import ORTH, LEMMA, POS
import re
import en_core_web_md as spacyEn
import spacy 
from spacy.symbols import nsubj, VERB, PROPN, dobj, prep, agent, nsubjpass, attr, conj, neg, aux
import os 
from os import listdir
from os.path import isfile, join


# extract other files 
os.chdir('C:/Users/sankisar/Documents/AI/Projects/Default Automation/Code')
from produce_extractions_functions import * 
os.chdir('C:/Users/sankisar/Documents/AI/Projects/Default Automation/')



# javapath needs to be set if one cannot set it through environment variables 
java_path = "C:/Program Files (x86)/Java/jre1.8.0_121/bin/java.exe"
os.environ['JAVAHOME'] = java_path

home = 'C:/Users/sankisar/Documents/AI/Stanford Core NLP'


"""  -- POS Tagging -- """
_path_to_model = home + '/stanford-postagger/models/english-bidirectional-distsim.tagger' 
_path_to_jar = home + '/stanford-postagger/stanford-postagger.jar'
st = POS_Tag(_path_to_model, _path_to_jar)

text = 'Yeshivah Ohel Moshe filed for chapter 11 bankruptcy protection Bankr. E.D.N.Y. Case No. 16-43681 on August 16 2016.'
st.tag(text.split())
# POS tagging works 

"""  -- Named Entity Recognition -- """
# for Named Entity recognition 
_path_to_model = home + '/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'
_path_to_jar = home + '/stanford-ner/stanford-ner-3.8.0.jar'

st = StanfordNERTagger(_path_to_model, _path_to_jar)
st.tag(text.split())

# NER works 

""" -- Neural Network Based Dependency Parsing -- """
os.environ['STANFORD_PARSER'] = home + '/stanford-parser/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = home + '/stanford-parser/stanford-parser-3.8.0-models.jar'

# we need to extract these by running following command on COMMAND PROMPT from the 
# directory where the stanford-parser-3.8.0-models.jar is located 
# python -mzipfile -e stanford-parser-3.8.0-models.jar models
# there is other method. Please see below link 
#https://stackoverflow.com/questions/11850574/python-unpacking-a-jar-file-doesnt-work
# https://stackoverflow.com/questions/13883277/stanford-parser-and-nltk


#_path_to_model = home + "/stanford-parser/models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"   
_path_to_model = home + "/stanford-corenlp/models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"   
#_path_to_model = home + "/stanford-parser/models/edu/stanford/nlp/models/parser/nndep/english_SD.gz"    
# lexical parser    
parser = stanford.StanfordParser(_path_to_model)
list(parser.raw_parse(text))
sentences = parser.raw_parse_sents(text)

# dependency parser
dep_parser=StanfordDependencyParser(_path_to_model)
# list(dep_parser.raw_parse(text))
print([parse.tree() for parse in dep_parser.raw_parse(text)])
for parse in dep_parser.parse(section.split()):
    parse.tree().draw()

# neural network based parser 
_path_to_model = home + "/stanford-corenlp/models/edu/stanford/nlp/models/parser/nndep/english_SD.gz"
parser = stanford.StanfordParser(_path_to_model)
list(parser.raw_parse(text))
for tree in parser.parse(section.split()):
    print(tree)
    tree.draw()

#NPs = list(tree.subtrees(filter=lambda x: x.label()=='NP'))
#NNs_inside_NPs = map(lambda x: list(x.subtrees(filter=lambda x: x.label()=='NNP')), NPs)
#[noun.leaves()[0] for nouns in NNs_inside_NPs for noun in nouns]
#ROOT = 'NP'
#def getNodes(parent):
#    for node in parent:
#        if type(node) is nltk.Tree:
#            if node.label() == ROOT:
#                print(" ".join(node.leaves()))
#                return(" ".join(node.leaves()))
#            getNodes(node)
#        else:
#            print("Word:", node)

#getNodes(tree)

#NEED TO WORK ON EXTRACTING TREE LEAVES AND NODES 
#tree = parser.parse(text.split())
# getNodes(tree)


###########################
# WORKING WITH SPACY 
###########################

nlp =  spacyEn.load()

text = 'On August 16 2016 Yeshivah Ohel Moshe filed for chapter 11 bankruptcy protection Bankr. E.D.N.Y. Case No. 16-43681 on August 16 2016.'
text = "Powell Valley Health Care Inc. provides healthcare services to the greater-Powell Wyoming community.  The Company filed for Chapter 11 bankruptcy protection Bankr. D. Wyo. Case No. 16-20326 on May 16 2016.  The petition was signed by Michael L. Long CFO.  The case is assigned to Judge Cathleen D. Parker.  The Debtor estimated assets and debts at 10 million to 50 million at the time of the filing."

doc = nlp(text, parse = True)

doc = nlp(text)
for word in doc:
    print(word.text, word.pos_, word.dep_, word.head.text)

root = [w for w in doc if w.head is w][0]


for word in doc:
    print(word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_, word.head.text, word.dep_)
    
    
[(token, token.label_) for token in doc.ents]


for ent in doc:
    if ent.dep == nsubj and ent.head.pos == VERB:
        print(ent.text)


for possible_verb in doc:
    if possible_verb.pos == VERB:
        for possible_subject in possible_verb.children:
            print('verb: {}, subject {}:'.format(possible_verb, [item for item in possible_subject.lefts] + [possible_subject]))


# good implementation 
entity = []
for possible_verb in doc:
    if possible_verb.pos == VERB:
        for possible_subject in possible_verb.children:
            if possible_subject.dep == nsubj:
                for descendant in possible_subject.subtree:
                    entity.append(descendant.text)
' '.join(entity)            
            
            
root = [w for w in doc if w.head is w][0]
subject = list(root.lefts)[1]
for descendant in subject.subtree:
    print(descendant)



###############################
# SOME CUSTOM FUNCTIONS 
############################### 
"""
Assumptions: 
    - Assume that all sentences are constructed in active voice. Passive voice will be supported in 
    next upgrade.  
    - Coreference resolution is addressed to reasonable degree. Only upto two sentences. 
    
General strategy: 
    - receive a text paragraph. 
    - break it into pieces and parse using Arc dependency structure. 
    - remove prepositions after the verbs that we are interested in. 
    - Look up for nsubj and dobj parts of speech. 
    - if we see nsubj as Debtor or the Company, look up for main nsubj entity in 
    - the preceding sentence and use that. (this is where active voices comes to play)
    - try and answer the question of what the company does based on the text! 

We will use Spacy tools almost exclusively here.  
    
"""

# use this example all the way 
text = "Powell Valley Health Care Inc. provides healthcare services to the greater-Powell Wyoming community.  The Company filed for Chapter 11 bankruptcy protection Bankr. D. Wyo. Case No. 16-20326 on May 16 2016.  The petition was signed by Michael L. Long CFO.  The case is assigned to Judge Cathleen D. Parker.  The Debtor estimated assets and debts at 10 million to 50 million at the time of the filing."
remove_words = ['Bankr', 'bankr']
default_verbs = ['filed', 'file','sought','files']
remove_subjects = ['The Company', 'The Debtor', 'The debtor','The company', 'She','He','she','he','It','it']


def summarize_paragraph(text, nlp = nlp, 
                        remove_words = remove_words, 
                        default_verbs = default_verbs, 
                        remove_subjects = remove_subjects):
    """ recognizes named entities in a piece of text. """
    
    # remove case numbers using regular expressions 
    s = re.sub(r'Bankr.+\d\d-\d\d\d\d\d.','',text)
    s = re.sub(r'--',',',s)
    s = re.sub(r'-','_',s)
    s = re.sub(r'Inc.', 'Inc',s)
    doc = nlp(s, parse = True)
    sentences = []
    subjects = []
    objects = []
    verbs = [] 
    part_sent = []
    recon_para = ''
    for sent in doc.sents:
        # check if the sentence has verb. if not, ignore the sentence 
        if len([w for w in sent if w.pos == VERB and w.text not in remove_words]) == 0:
            continue
        for possible_verb in sent:
            sent_verbs = []
            sent_subjects = []
            sent_objects = []
            if possible_verb.pos == VERB:
                if possible_verb.dep_ == 'ROOT' and possible_verb.text[0].islower(): 
                    sent_verbs.append(possible_verb.text)
                elif possible_verb.dep_ != 'ROOT' and possible_verb.text in default_verbs:
                    sent_verbs.append(possible_verb.text)
                elif possible_verb.dep_ == 'ROOT' and possible_verb.text[0].isupper():
                    continue
                else:
                    continue
                #sent_verbs.append(possible_verb.text)
                for possible_subject in possible_verb.children:
                    if possible_subject.dep == nsubj or possible_subject.dep == nsubjpass or possible_subject.dep == neg or possible_subject.dep == aux:
                        for descendant in possible_subject.subtree:
                            sent_subjects.append(descendant.text)
                    if possible_subject.dep == dobj or possible_subject.dep == prep or possible_subject.dep == agent or possible_subject.dep == attr:
                        for descendant in possible_subject.subtree:
                            sent_objects.append(descendant.text)
                            
                sent_objects = [w for w in sent_objects if w not in remove_words]
                part_sent = sent_subjects + sent_verbs + sent_objects
                sentences.append(' '.join(part_sent))
                subjects.append(' '.join(sent_subjects))
                verbs.append(' '.join(sent_verbs))
                objects.append(' '.join(sent_objects))
                
        # remove stop words from the list of words collected 
        sent_objects = [w for w in sent_objects if w not in remove_words]
        # if we see two verbs, we need to prioritize default verbs 
        
        # if we have multiple verbs are present in the same sentence 
#        subjects.append(' '.join(sent_subjects))
#        verbs.append(' '.join(sent_verbs))
#        objects.append(' '.join(sent_objects))
#        sentence = ' '.join(sent_subjects) + ' ' + ' '.join(sent_verbs) + ' ' +  ' '.join(sent_objects) 
#        # reconstruct the paragraph 
#        sentences.append(sentence)
    recon_para = '. '.join(sentences)
    
    """ at this point, we have summarized the paragraph, now, 
    let's look at identifying defaulted entities """
    defaulted_entity = "None Found"
    for i, v in enumerate(verbs):
        if v in default_verbs and i == 0:
            defaulted_entity = subjects[0]
            break 
        elif v in default_verbs:
            # at this point, we will return the subject from the first useful sentence in the paragraph 
            if subjects[i] not in remove_subjects:
                defaulted_entity = subjects[i]
                break
            else:
                defaulted_entity = [ent for ent in subjects[:i] if ent not in remove_subjects][0]
            break
        else:
            defaulted_entity = 'None Found'
    # reshaping 
    defaulted_entity = re.sub(r'_','-', defaulted_entity)
    recon_para = re.sub(r'_','-',recon_para)
    sentences = [re.sub(r'_','-',sent) for sent in sentences]
    subjects = [re.sub(r'_','-',subj) for subj in subjects]
    objects = [re.sub(r'_','-',obj) for obj in objects]
    # return output 
    return defaulted_entity, recon_para, sentences, subjects, verbs, objects
        


# testing 
text = 'Yeshivah Ohel Moshe filed for chapter 11 bankruptcy protection Bankr. E.D.N.Y. Case No. 16-43681 on August 16 2016.'

#text = sections[0]

text = 'XXX works for Moody Analytics. He advises clients. The debtor filed for bankruptcy. XA was appointed as a judge'
text = 'Lewisville Texas-based ADPT DFW Holdings LLC Bankr. N.D. Tex. Case No. 17-31432 and its affiliates each filed separate Chapter 11 bankruptcy petitions on April 19 2017 listing 798.67 million in total assets and 453.48 million in total debts as of Sept. 30'
text = 'Lewisville Texas-based ADPT DFW Holdings LLC and its affiliates each filed separate Chapter 11 bankruptcy petitions on April 19 2017 listing 798.67 million in total assets and 453.48 million in total debts as of Sept. 30'
text = 'Advanced Biomedical Inc. dba Pathology Laboratories Services Inc. filed a Chapter 11 petition Bankr. C.D. Calif. Case No. 14-15938 on October 1 2014 and is represented by Robert Sabahat Esq. at Madison Harbor ALC in Irvine California.  At the time of its filing the Debtors estimated assets was 100000 to 500000 and estimated liabilities was 1 million to 10 million.  The petition was signed by Cyrus Karimi president.  The Debtor did not file a list of its largest unsecured creditors when it filed the petition.'




# build a regular expression parser to remove case number etc., 
#text = ' '.join([w.replace('Bankr.','bankr').replace('N.D.','').replace('S.D.','') for w in text.split()])
#text = ' '.join([w.replace('Inc.','Incorporated')for w in text.split()])


ent, recon_para, sentences, subjects,verbs,objects = summarize_paragraph(text, nlp)  
ent
    
    

# missing ones # 
text = 'Going Ventures LLC which operates under the name Going Aire LLC filed a Chapter 11 petition Bankr. S.D. Fla. Case No. 17-12747 on March 7 2017.  Carl Bradley Copeland manager signed the petition.  Judge Laurel M. Isicoff is the case judge.  The Debtor is represented by David R. Softness Esq. of David R. Softness P.A.  At the time of filing the Debtor had total assets of 72900 and total liabilities of 1.01 million.  No trustee examiner or statutory committee has been appointed in the Debtors Chapter 11 case.'
text = 'On-Call Staffing Inc. filed a Chapter 11 petition Bankr. N.D. Miss. Case No. 16-13823 on Oct. 28 2016.  The Debtor is represented by J. Walter Newman IV Esq. at Newman  Newman. The petition was signed by its President Lee Garner III.  At the time of the filing the Debtor estimated assets at 100000 to 500000 and liabilities at 500000 to 1 million.'
text = 'R.E.S. Nation LLC filed a Chapter 11 petition Bankr. S.D. Tex. Case No. 16-34744 on Sept. 23 2016. The petition was signed by Jeffrey Nowling manager. The Debtor tapped Susan C. Matthews Esq. at Baker Donelson Bearman Caldwell  Berkowitz APC. At the time of filing the Debtor estimated assets and liabilities at up to 50000.'





text = 'XXX works for Moody Analytics. He advises clients. The Debtor filed for bankruptcy. XA was appointed as a judge.'
text = "The councilmen refused the demonstrators a permit because they feared violence"
doc = nlp(s, parse = True)

#sentences = [sent for sent in doc.sents]
#s = re.sub(r'Bankr.+No\.','',text)
#s = re.sub(r'Bankr.+\d\d-\d\d\d\d\d.','',text)


# story examples 
text = "Powell Valley Health Care Inc. provides healthcare services to the greater-Powell Wyoming community.  The Company filed for Chapter 11 bankruptcy protection Bankr. D. Wyo. Case No. 16-20326 on May 16 2016.  The petition was signed by Michael L. Long CFO.  The case is assigned to Judge Cathleen D. Parker.  The Debtor estimated assets and debts at 10 million to 50 million at the time of the filing."
text = 'On-Call Staffing Inc. filed a Chapter 11 petition Bankr. N.D. Miss. Case No. 16-13823 on Oct. 28 2016.  The Debtor is represented by J. Walter Newman IV Esq. at Newman  Newman. The petition was signed by its President Lee Garner III.  At the time of the filing the Debtor estimated assets at 100000 to 500000 and liabilities at 500000 to 1 million.'
text = 'Advanced Biomedical Inc. dba Pathology Laboratories Services Inc. filed a Chapter 11 petition Bankr. C.D. Calif. Case No. 14-15938 on October 1 2014 and is represented by Robert Sabahat Esq. at Madison Harbor ALC in Irvine California.  At the time of its filing the Debtors estimated assets was 100000 to 500000 and estimated liabilities was 1 million to 10 million.  The petition was signed by Cyrus Karimi president.  The Debtor did not file a list of its largest unsecured creditors when it filed the petition.'


text = "In Houston, after the first night of a citywide curfew, many residents went outside for the first time in days to survey the wreckage. Officials have reported at least 38 deaths."
text = "Using a fleet of Cajun-style airboats, Jet Skis and fishing boats, a massive volunteer rescue effort patrolled the roads turned rivers."
text = "Thousands are applying for federal assistance, but it may be slow to arrive and require them to take on debt that could take years to pay off."
text = "After the New America Foundation praised a large fine levied on Google, the man behind the statement was fired."
text = "Pondering California’s regions, a lingering heat wave, and a look back at Ishi, the last known survivor of the Yahi tribe."
text = "Amazon’s Alexa and Microsoft’s Cortana Can Soon Talk to Each Other"

































































































