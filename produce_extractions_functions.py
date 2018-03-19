# -*- coding: utf-8 -*-
"""
This file provides functions to read text from docx and txt files and subsequently
search through these files for pre-specified lookup words.

Created on Tue Aug  8 13:22:56 2017
@author: SankisaR
"""
# import libraries 
import os 
from docx import Document
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tag import StanfordNERTagger
import re
import en_core_web_md as spacyEn
import spacy 
from spacy.symbols import nsubj, VERB, PROPN, dobj, prep, agent, nsubjpass, attr, conj, neg, aux
java_path = "C:/Program Files (x86)/Java/jre1.8.0_121/bin/java.exe"
os.environ['JAVAHOME'] = java_path


"""
This code processes and extracts useful information on defaults using the 
split docx files. 

"""
# set working directory 
#os.chdir('C:/Users/sankisar/Documents/AI/Projects/Default Automation/')

"""
functions:
    - produce_n_gram: produces n_grams from word tokens. expects list of words
    - getTextFromWordFile: read word files and outputs a text
    - getTextFromTextFile: read from .txt files 
    - date_extractor: extracts date from a text using POS tags 
    - binary_highlighter: the function highlights presence of lookup words in supplied text

"""
# create a function for n-gram 
def produce_n_gram(inp, n_gram = 2, delimiter = "_"):
    """
    Function to produce n_grams for text mining. 
    This function does not use the part-of-speech tags
    This just produces the required n-grams 
    if the n_gram count is lower than the number of words in the 
    sentence, the function would return a null list 
    
    """
    # check if the len of input sentence is less than n_gram
    # return null list in that case 
    if len(inp) < n_gram:
        return []
    else:
        out = [delimiter.join(inp[i:(n_gram + i)]) for i in range(len(inp)-n_gram + 1)]
        return out
    

# extract text from docx files   
def getTextFromWordFile(filename):
    """
    Extract text from docx file. Function taken with minimal changes from 
    https://stackoverflow.com/questions/25228106/how-to-extract-text-from-an-existing-docx-file-using-python-docx
    We will use this text as a starting point from extractions

    """
    doc = Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


def getTextFromTextFile(filename):
    """
    Extracts text from text file 
    """
    text = ''
    with open(filename, 'r') as f:
        for line in f:
            text = text + line
    
    return text
    

def date_extractor(section):
    """
    this function extracts date from a section that has some default relevance. 
    We use POS tags to identify specific parts that point to a date
    
    This approach makes it agnostic to different types of specification on 
    date formats. 
    
    Relies on nltk.pos_tag function. Make sure to import nltk package  
    
    input argument 'section' expects a text string. 
    
    """
    lookup_pos_sequences = [('NNP_CD_CD'),
                            ('NNP_CD_,_CD'),
                           ('NNP_CD_._CD'),
                           ('NNP_CD_CD_.')]
    
    # get the part-of-speech of the sentence 
    word_pos_tags = nltk.pos_tag(nltk.word_tokenize(section))
    pos_tags = tuple(tag[1] for tag in word_pos_tags)
    words = tuple(tag[0] for tag in word_pos_tags)
    # now loop over the elements of this list 
    n_gram_tags = produce_n_gram(pos_tags, 3) + produce_n_gram(pos_tags, 4) 
    n_gram_words = produce_n_gram(words, 3, ' ') + produce_n_gram(words, 4, ' ')
    
    for seq in lookup_pos_sequences:
        if seq in n_gram_tags:
            date = n_gram_words[n_gram_tags.index(seq)]
            return date.replace(' .', '')
        else:
            continue
        
    return ''


def binary_highlighter(text, lookup_words, use_n_grams = (2,3), default_look_up_limit = 8):
    """
    This function highlights whether any of the lookup words exists. 
    if exists, it will return a flag 'True', else 'False'
    In addition it will also give the lookup word that we found
    in the input text. 
    
    This function is written with binary classification problem in mind.
    This function has following dependencies
    - sklearn CountVectorizer() class
    - string punctuation
    
    default_look_up_limit : function will look for default dates until these 
    lookup values in the lookup_list. If there are some key words which are not 
    generally associated with dates, you should add them to lookup list after this 
    number or adjust this number accordingly. 
    
    We will first split the file into paragraphs and look for specific 
    lookup words. We will return the paragraph that has the key words
    
    """
    # first strip the white spaces in the text 
    text = text.strip()
    paragraphs = text.split('\n\n')
    
    # create a list for all the lookup words caught in the text 
    out_list = []
    out_dict = {}
    flag = False 
    # create flag
  
    # to handle the punctuation: include comma. commented version removes comma 
    table = str.maketrans({ch: None for ch in '!"#$%&\'()*+/:;<=>?@[\\]^_`{|}~'})
    #table = str.maketrans({ch: None for ch in '!"#$%&\'()*+/,:;<=>?@[\\]^_`{|}~'})
    
    for para in paragraphs:
        # split the para into
        recon_para = [s.translate(table) for s in para.split('\n') if s != '']
        recon_para = ' '.join(recon_para)
        word_tokens = [w.lower() for w in CountVectorizer().build_tokenizer()(recon_para)]
        # we have the word tokens now
        # now create n-grams 
        n_gram_vec = word_tokens
        for n_gram in use_n_grams:
            n_gram_vec = n_gram_vec + produce_n_gram(word_tokens, n_gram)
        
        # now lookup words
        for lw in lookup_words:
            if lw in n_gram_vec:
                if lw not in out_list:
                    out_list.append(lw)
                    out_dict[lw] = []
                    if lw in lookup_words[:default_look_up_limit]:
                        event_date = date_extractor(recon_para)
                        out_dict[lw].append([recon_para, event_date])
                    else:
                        out_dict[lw].append(recon_para)
                else:
                    if lw in lookup_words[:default_look_up_limit]:
                        event_date = date_extractor(recon_para)
                        out_dict[lw].append([recon_para, event_date])
                    else:
                        out_dict[lw].append(recon_para)
        
        if len(out_list)!=0:
            flag = True
        
    return flag, out_dict, out_list


def flag_TCR_reports(filepath, filename, lookup_words, use_n_grams = (2,3), default_look_up_limit = 8):
    """
    Combines text extractor and binary_highlighter functions 
    checks whether the supplied file is text or docx first and calls appropriate function 
    Please note that the function only expects docx or txt files 
    
    """
    # check the doc type 
    if filename.split(".")[1] == 'docx':
        text = getTextFromWordFile(filepath + filename)
    elif filename.split(".")[1] == 'txt':
        text = getTextFromTextFile(filepath + filename)
    else:
        raise NotImplementedError()
    
    # call the binary classification function     
    return binary_highlighter(text, lookup_words, use_n_grams, default_look_up_limit)
        
# good working version 
def named_entity_recognizer_base(section):
    """
    this function extracts all the entities in the section. 
    The logic we follow is as follows:
        - look for all NNP POS which have first word capitalized
        - for numbers in between NNP, include them extraction of entity  
    we use simple POS tagger to determine the entities.
    entities can have arbitrary length of words. 
    We do not use CountVectorizer here since it removes numbers automatically 
    """
    # first split the section into tokens
    # first try and remove any dates that are in the document 
    filing_words = ['filed', 'file']
    date = date_extractor(section)
    if date is not "":
        if 'on ' + date in section:
            updated_section = section.replace('on ' + date, '')
        elif 'On ' + date in section:
            updated_section = section.replace("On " + date, "")
        else:
            updated_section = section.replace(date, '')
    else:
        updated_section = section[:]
        
    split_words = [w.replace('.','') for w in updated_section.split() if w is not "."]
    
    word_pos_tags = nltk.pos_tag(split_words)
    pos_tags = tuple(tag[1] for tag in word_pos_tags)
    words = tuple(tag[0] for tag in word_pos_tags)
    
    # pick all NNP before each VBD or other such verb form 
    first_verb_ind = None 
    try:
        first_verb_ind = min([pos_tags.index('VBD')] + [words.index(w) for w in filing_words if w in words])
    except ValueError:
        try:
            if pos_tags.index('VBN') - pos_tags.index('VBZ') == 1:
                first_verb_ind = pos_tags.index('VBZ')
            else:
                inds = [i for i, x in enumerate(pos_tags) if x == 'VBN']
                for ind in inds:
                    if pos_tags[ind - 1] == 'VBZ':
                        first_verb_ind = ind
                        break
                if first_verb_ind == None:
                    return 'No entity found' 
        except ValueError:
            return ' '

   
    entity_pos = pos_tags[:first_verb_ind] 
    # if we see a determiner in between NNPs just take the last one 
    if 'DT' in entity_pos:
        det_idx = len(entity_pos) - list(reversed(entity_pos)).index('DT') - 1
        if words[det_idx]  == 'a':
            # if we see 'a' as a determiner, we can look back 
            det_idx = -1
    else:
        det_idx = -1
    # get entity words 
    #entity_words = [words[det_idx + 1 + i] for i, tag in enumerate(pos_tags[det_idx+1:first_verb_ind]) 
    #if tag == 'NNP' or tag == 'CD']
    
    entity_words = []
    for i, tag in enumerate(pos_tags[det_idx+1:first_verb_ind]):
        """much easier to manage than the one-line implementation"""
        if (tag == 'NN' or tag == "NNPS") and words[det_idx+1+i][0].isupper():
            entity_words.append(words[det_idx+1+i])
        elif tag == "NNP" or tag == 'CD':
            entity_words.append(words[det_idx+1+i])
        elif tag == 'JJ' and  words[det_idx+1+i][0].isupper():
            entity_words.append(words[det_idx+1+i])
   
    # remove months which are usually tagged as NNP 
    trim_words = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december',
              'jan.','feb.', 'mar.', 'apr.', 'jun.','jul.', 'aug.', 'sep.',
              'oct.','nov.', 'dec.','jan','feb', 'mar', 'apr', 'jun','jul', 
              'aug', 'sep','oct','nov', 'dec']
    # trim words 
    entity_words = [w for w in entity_words if w.lower() not in trim_words]
    entity = ' '.join(entity_words)

    return entity    



def named_entity_recognizer(section):
    """
    this function extracts all the entities in the section. 
    The logic we follow is as follows:
        - look for all NNP POS which have first word capitalized
        - for numbers in between NNP, include them extraction of entity  
    we use simple POS tagger to determine the entities.
    entities can have arbitrary length of words. 
    We do not use CountVectorizer here since it removes numbers automatically 
    """
    # first split the section into tokens
    # first try and remove any dates that are in the document 
    filing_words = ['filed', 'file', 'sought']
    purge_after_words = ['Inc', 'Incorporated', 'Corporation','LLC']
    date = date_extractor(section)
    if date is not "":
        if 'on ' + date in section:
            updated_section = section.replace('on ' + date, '')
        elif 'On ' + date in section:
            updated_section = section.replace("On " + date, "")
        else:
            updated_section = section.replace(date, '')
    else:
        updated_section = section[:]
        
    split_words = [w.replace('.','') for w in updated_section.split() if w is not "."]
    
    word_pos_tags = nltk.pos_tag(split_words)
    pos_tags = tuple(tag[1] for tag in word_pos_tags)
    words = tuple(tag[0] for tag in word_pos_tags)
    
    # pick all NNP before each VBD or other such verb form 
    first_verb_ind = None 
    try:
        #first_verb_ind = min([pos_tags.index('VBD')] + [words.index(w) for w in filing_words if w in words])
        first_verb_ind = min([words.index(w) for w in filing_words if w in words])
    except ValueError:
        try:
            if pos_tags.index('VBN') - pos_tags.index('VBZ') == 1:
                first_verb_ind = pos_tags.index('VBZ')
            else:
                inds = [i for i, x in enumerate(pos_tags) if x == 'VBN']
                for ind in inds:
                    if pos_tags[ind - 1] == 'VBZ':
                        first_verb_ind = ind
                        break
                if first_verb_ind == None:
                    return 'No entity found' 
        except ValueError:
            return ' '
    entity_pos = list(pos_tags[:first_verb_ind])
    # remove the determiner if it happens to be the last POS 
    if entity_pos[-1] == 'DT':
        del entity_pos[-1]
    # if we see a determiner in between NNPs just take the last one 
    if 'DT' in entity_pos:
        det_idx = len(entity_pos) - list(reversed(entity_pos)).index('DT') - 1
        if words[det_idx]  == 'a':
            # if we see 'a' as a determiner, we can look back 
            det_idx = -1
    else:
        det_idx = -1
    # get entity words 
    #entity_words = [words[det_idx + 1 + i] for i, tag in enumerate(pos_tags[det_idx+1:first_verb_ind]) 
    #if tag == 'NNP' or tag == 'CD']
    entity_words = []
    for i, tag in enumerate(pos_tags[det_idx+1:first_verb_ind]):
        """much easier to manage than the one-line implementation"""
        if (tag == 'NN' or tag == "NNPS") and words[det_idx+1+i][0].isupper():
            entity_words.append(words[det_idx+1+i])
        elif tag == "NNP" or tag == 'CD':
            entity_words.append(words[det_idx+1+i])
        elif tag == 'JJ' and  words[det_idx+1+i][0].isupper():
            entity_words.append(words[det_idx+1+i])
   
    # remove months which are usually tagged as NNP 
    trim_words = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december',
              'jan.','feb.', 'mar.', 'apr.', 'jun.','jul.', 'aug.', 'sep.',
              'oct.','nov.', 'dec.','jan','feb', 'mar', 'apr', 'jun','jul', 
              'aug', 'sep','oct','nov', 'dec','Bankr','ND','Tex','Case','No']
    # trim words 
    entity_words = [w for w in entity_words if w.lower() not in trim_words]
    # purge all words after inc
    for w in purge_after_words:
        if w in entity_words:
            idx = entity_words.index(w)
            del entity_words[idx+1:]
    entity = ' '.join(entity_words)
    # we can check the length of entity and see if we are outputting meaningless 
    # entity names 
    return entity    



def stanford_named_entity_recognizer(section, tagger_object):
    """ 
    This function uses stanford NER tagger that comes with NLTK. 
    This requires few additional settings for it to work
    """
    filing_words = ['filed', 'file', 'sought']
    split_words = [w.replace('.','') for w in section.split() if w is not "."]
    idx = len(split_words) 

    for i, w in enumerate(split_words):
        for wf in filing_words:
            if w == wf:
                idx = i
                break
    # we use words only before the key verb. 
    out = tagger_object.tag(split_words[:idx+1])
    ner = []
    for item in out:
        if item[1] == 'ORGANIZATION' and item[0] not in ner:
            ner.append(item[0])
    return ' '.join(ner)
    
   
def spacy_named_entity_recognizer(text, nlp, 
                        remove_words = [], 
                        default_verbs = ['filed', 'file','sought','files','seeks', "commenced", "commence"], 
                        remove_subjects = ['The Company', 'The Debtor', 'The debtor',
                                           'The company', 'She','He','she','he','It','it', 
                                           'They','they']):
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
    #recon_para = ''
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
                    if possible_subject.dep == nsubj or possible_subject.dep == nsubjpass:
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
                #defaulted_entity = [ent for ent in subjects[:i] if ent not in remove_subjects][0]
            break
        else:
            defaulted_entity = 'None Found'
    # reshaping 
    defaulted_entity = re.sub(r'_','-', defaulted_entity)

    return defaulted_entity



    
def summarize_paragraph(text, nlp, 
                        remove_words = [], 
                        default_verbs = ['filed', 'file','sought'], 
                        remove_subjects = ['The Company', 'The Debtor', 'The debtor',
                                           'The company', 'She','He','she','he','It','it', 
                                           'They','they']):
    """ recognizes named entities in a piece of text. """
    
    # remove case numbers using regular expressions 
    s = re.sub(r'Bankr.+\d\d-\d\d\d\d\d.','',text)
    s = re.sub(r'--',',',s)
    s = re.sub(r'-','_',s)
    s = re.sub(r'Inc.', 'Inc',s)
    doc = nlp(s)
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
                #defaulted_entity = [ent for ent in subjects[:i] if ent not in remove_subjects][0]
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

            
        
                
                
            

        
    
    
    










































    










