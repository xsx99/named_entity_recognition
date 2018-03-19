# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:01:38 2017

@author: XuL
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import sys
#sys.setdefaultencoding("utf-8")

import nltk
from nltk.tree import *
import nltk.data
import nltk.draw
from nltk import tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import re
import pandas as pd
#import textacy

# Stanford Parser
os.environ['STANFORD_PARSER'] = 'C:\\Users\\XuL\\stanford\\stanford-parser-full-2017-06-09\\stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'C:\\Users\\XuL\\stanford\\stanford-parser-full-2017-06-09\\stanford-parser-3.8.0-models.jar'
from nltk.parse import stanford


# Ghostscript for displaying of tree structures
path_to_gs = "C:\\Program Files\\gs\\gs9.22\\bin"
os.environ['PATH'] += os.pathsep + path_to_gs


# Java path
java_path = "C:/Program Files (x86)/Java/jre1.8.0_152/bin"
os.environ['JAVAHOME'] = java_path


# Stanford NER
_model_filename = 'C:\\Users\\XuL\\AppData\\Local\\Continuum\\anaconda3\\stanford_ner\\classifiers\\english.all.3class.distsim.crf.ser.gz'
_path_to_jar = 'C:\\Users\\XuL\\AppData\\Local\\Continuum\\anaconda3\\stanford_ner\\stanford-ner.jar'
from nltk.tag.stanford import StanfordNERTagger
st = StanfordNERTagger(model_filename=_model_filename, path_to_jar=_path_to_jar)


# Spacy NER
import spacy
import en_core_web_md as spacyEn



global nlp
nlp =  spacyEn.load()




class SVO(object):
    
    
    """
    Class Methods to Extract Subject Verb Object Tuples from a Sentence
    """
    def __init__(self):
        """
        Initialize the SVO Methods
        """
        #self.noun_types = ["NN", "NNP", "NNPS","NNS","PRP","CC","CD"]
        self.verb_types = ["VB","VBD","VBG","VBN", "VBP", "VBZ"]
        self.noun_types = ["NP"]
        #self.verb_types = ["VP"]
        self.adjective_types = ["JJ","JJR","JJS"]
        self.pred_verb_phrase_siblings = None
        self.parser = stanford.StanfordParser()
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')






    def get_attributes(self,node,parent_node, parent_node_siblings):
        """
        returns the Attributes for a Node
        """






    def get_subject(self,sub_tree):
        """
        Returns the Subject and all attributes for a subject, sub_tree is a Noun Phrase
        """
        sub_nodes = []
        sub_nodes = sub_tree.subtrees()
        sub_nodes = [each for each in sub_nodes if each.pos()]
        subject = None

        for each in sub_nodes:

            if each.label() in self.noun_types:
                subject = each.leaves()
                break

        return {'subject':subject}






    def get_object(self,sub_tree):
        """
        Returns an Object with all attributes of an object
        """
        siblings = self.pred_verb_phrase_siblings
        Object = None
        for each_tree in sub_tree:
            if each_tree.label() in ["NP","PP"]:
                sub_nodes = each_tree.subtrees()
                sub_nodes = [each for each in sub_nodes if each.pos()]

                for each in sub_nodes:
                    if each.label() in self.noun_types:
                        Object = each.leaves()
                        break
                break
            else:
                sub_nodes = each_tree.subtrees()
                sub_nodes = [each for each in sub_nodes if each.pos()]
                for each in sub_nodes:
                    if each.label() in self.adjective_types:
                        Object = each.leaves()
                        break
                # Get first noun in the tree
        self.pred_verb_phrase_siblings = None
        return {'object':Object}






    def get_predicate(self, sub_tree):
        """
        Returns the Verb along with its attributes, Also returns a Verb Phrase
        """

        sub_nodes = []
        sub_nodes = sub_tree.subtrees()
        sub_nodes = [each for each in sub_nodes if each.pos()]
        predicate = None
        pred_verb_phrase_siblings = []
        sub_tree  = ParentedTree.convert(sub_tree)
        for each in sub_nodes:
            if each.label() in self.verb_types:
                sub_tree = each
                predicate = each.leaves()

        #get all predicate_verb_phrase_siblings to be able to get the object
        sub_tree  = ParentedTree.convert(sub_tree)
        if predicate:
             pred_verb_phrase_siblings = self.tree_root.subtrees()
             pred_verb_phrase_siblings = [each for each in pred_verb_phrase_siblings if each.label() in ["NP","PP","ADJP","ADVP"]]
             self.pred_verb_phrase_siblings = pred_verb_phrase_siblings

        return {'predicate':predicate}





    def process_parse_tree(self,parse_tree):
        """
        Returns the Subject-Verb-Object Representation of a Parse Tree.
        Can Vary depending on number of 'sub-sentences' in a Parse Tree
        """
        self.tree_root = parse_tree
        # Step 1 - Extract all the parse trees that start with 'S'
        svo_list = [] # A List of SVO pairs extracted
        output_list = []


        

        for idx, subtree in enumerate(parse_tree[0].subtrees()):
            output_dict ={}
            subject =None
            predicate = None
            Object = None
            if subtree.label() in ["S", "SQ", "SBAR", "SBARQ", "SINV", "FRAG"]:
                children_list = subtree
                children_values = [each_child.label() for each_child in children_list]
                children_dict = dict(zip(children_values,children_list))

                keys = ['file','seek','default','estimate','commence']

                
                # only extract nountype words from sentences contain keywords
                if children_dict.get("NP") is not None:
                    v = children_dict.get("VP")
                    if  v is not None:
                        leaves = v.leaves()
                        leaves = [lemmatizer.lemmatize(leaf,'v') for leaf in leaves]
                        haskey = len([1 for leaf in leaves if leaf in keys])
                        if haskey > 0:
                            subject = self.get_subject(children_dict["NP"])
                            print(subject['subject'])
               # if children_dict.get("VP") is not None:
                    # Extract Verb and Object
                    #i+=1
                    #"""
                    #if i==1:
                    #    pdb.set_trace()
                    #"""
                   # predicate = self.get_predicate(children_dict["VP"])
                   # Object = self.get_object(children_dict["VP"])

                try:
                    if subject['subject']: #or predicate['predicate'] or Object['object']:
                        output_dict['subject'] = subject['subject']
                        #output_dict['predicate'] = predicate['predicate']
                        #output_dict['object'] = Object['object']
                        output_list.append(output_dict)
                except Exception:
                        continue



        return output_list





    def traverse(self,t):
        try:
            t.label()
        except AttributeError:
            print(t)
        else:
            # Now we know that t.node is defined
            print('(', t.label())
            for child in t:
                self.traverse(child)

            print(')')





# check if the input sentence tree includes a node with label "S"
    def check_clause(self, sent):    

        clause_tags = ['S', 'SBAR', 'SBARQ', 'SINV', 'SQ']        
        global result
        result = False
        def check_sent(t):
            global result
            try:
                if (t.label() in clause_tags) & (t.height()>1):
                    result = True
            except AttributeError:
                    pass
            else:
                 if (t.label() in clause_tags) & (t.height()>1):
                    result = True
                 for child in t:
                     check_sent(child)
        check_sent(sent)             
        return result





    def retrive_lowest_clauses(self, sent):
            clauses = []
            if not self.check_clause(sent):
                clauses += []
            else:
                try:
                    tmp = 0
                    for child in sent:
                        tmp += self.check_clause(child)
                    if tmp == 0:
                        clauses += [sent]
                    else:
                        for child in sent:
                            # clauses += child - S
                            clauses += self.retrive_lowest_clauses(child)
                # when reaching leaves            
                except TypeError: 
                    clauses += []
            return clauses






    def retrive_clauses(self, sent):
        
            sent = ParentedTree.convert(sent)
            clauses = []
            
            lowest_clauses = self.retrive_lowest_clauses(sent)
            
            while (lowest_clauses !=[]): 
                clauses += lowest_clauses
                for lowest_clause in lowest_clauses:
                    del sent[lowest_clause.treeposition()]
                    lowest_clauses = self.retrive_lowest_clauses(sent)            
                
            return clauses




    def find_case(self, x):    
        
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
    
    
    
    
    
    def remov_case(self, x):
        
        new_x = self.find_case(x)
        while new_x != x:
            x = new_x
            new_x = self.find_case(x)
            
        return new_x




    def pre_process(self,text):
        
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
      
        
        s = re.sub(r' Corp\.', ' Corporation', s)
        s = re.sub(r' Co\. ', ' Co ', s)
        s = re.sub(r' Dev\.', ' Dev', s)
        s = re.sub(r' Assoc\.', ' Association', s)
        s = re.sub(r'Mil\.', 'million', s)
        s = re.sub(r' Hon\. ', ' Hon ', s)
        s = re.sub(r' Ind\. ', ' Ind ', s)
        
        # convert numbers
#        s = s.replace(",000,000", " million").replace(",000,001", " million").\
#        replace(",000 ", " thousand").replace(",001", " thousand")
        
        # remove short forms
        s = s.replace("′", "'").replace("’", "'").\
        replace("won't", "will not").replace("cannot", "can not").\
        replace("can't", "can not").replace("n't", " not").\
        replace("what's", "what is").replace("'ve", " have").\
        replace("I'm", "I am").replace("'re", " are").\
        replace("%", " percent ").replace("$", " dollar ").\
        replace("'ll", " will").replace(" it's ", " its ")
 
        # remove bankruptcy case numbers
        s = self.remov_case(s)
        
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






    def sentence_split(self, text):
        """
        returns the Parse Tree of a Sample
        """
        doc = nlp(text)
        sentences = doc.sents
        return [str(sentence) for sentence in sentences]




    def sentence_split_2(self, text):
        return tokenize.sent_tokenize(text)




    def get_parse_tree(self,sentence):
        """
        returns the Parse Tree of a Sample
        """
        parse_tree = self.parser.raw_parse(sentence)

        return parse_tree









    def List_To_Tree(self,lst):
        if(not isinstance(lst, basestring)):
            if(len(lst) == 2 and isinstance(lst[0], basestring) and isinstance(lst[1], basestring)):
                lst = Tree(str(lst[0]).split('+')[0],[str(lst[1])])
            elif(isinstance(lst[0], basestring) and not isinstance(lst[1], basestring)):
                lst = Tree(str(lst[0]).split('+')[0],map(List_To_Tree, lst[1: len(lst) ]))
        return lst

    
    
    
    
    

    def  return_clause_list(self, sentences):
        
        clause_trees = []
        
        for sent in sentences:
            root_tree = next(self.get_parse_tree(sent))
            clause_trees += self.retrive_clauses(root_tree)
        
        clauses = []
        for cl in clause_trees:
            clauses.append(' '.join(cl.leaves()))
        
        return clauses
    


#if __name__=="__main__":
    
#    svo = SVO()
    

#    sentence = "David D. Zimmerman, Clerk of the U.S. Bankruptcy Court found Wells Fargos Motion to be defective as no opportunity to object to the entire matrix was filed."  
#    sentence = "Mr. Zimmerman refused to take action on Wells Fargos Motion and schedule a hearing, until Wells Fargo corrects the deficiency within 14 days from May 2, 2016, or May 16, 2016."  
#    sentence = "Headquartered in Mechanicsburg, Pennsylvania, Choudries Inc. dba Super Seven Food Mart filed for Chapter 11 bankruptcy protection Bankr. M.D. Pa. Case No. 16-02475 on June 13, 2016, and is represented by Gary J. Imblum, Esq., at Imblum Law Offices, P.C."  
#    sentence = "Headquartered in Long Island City, New Yok, Anthony Lawrence of New York, Inc., filed for Chapter 11 bankruptcy protection Bankr. E.D.N.Y. Case No. 15-44702 on Oct. 15, 2015, estimating its assets at up to 50,000 and its liabilities at between 1 million and 10 million. The petition was signed by Joseph J. Calagna, president. Judge Elizabeth S. Stong presides over the case. James P Pagano, Esq., who has an office in New York, serves as the Debtors bankruptcy counsel."
#    sentence = "On April 21, 2016, SunEdison, Inc., and 25 of its affiliates each filed a Chapter 11 bankruptcy petition Bankr. S.D.N.Y. Case Nos. 16-10991 to 16-11017.  Martin H. Truong signed the petitions as senior vice president, general counsel and secretary."    
#    sentence = "Based in Rochester, Michigan, TAJ Graphics Enterprises, LLC, filed for Chapter 11 bankruptcy protection Bankr. E.D. Mich. Case No. 09-72532 on Oct. 21, 2009.  John D. Hertzberg, Esq., in Bingham Farms, Michigan, serves as the Debtors counsel. In its petition, the Debtor estimated 10 million to 50 million, and 1 million to 10 million in debts."   
 
#    sentence = "The D rating reflects our expectation that Stone Energy will elect to file for Chapter 11 bankruptcy protection rather than make the May interest payment on its 7.5 senior unsecured notes due 2022, said SP Global Ratings credit analyst David Lagasse."
#    sentence = "Judge Robert Jacobvitz of the U.S. Bankruptcy Court in New Mexico denied the appointment of Mr. Pierce who was selected by the U.S. trustee overseeing Railyards bankruptcy case to replace the companys management."   

#    sentence = "The U.S. Trustee also appointed a Committee of Asbestos Creditors on April 28, 2000.  The Bankruptcy Court authorized the retention of these professionals by the Committee of Asbestos Creditors i Caplin  Drysdale, Chartered as Committee Counsel ii Campbell  Levine as local counsel iii Anderson Kill  Olick, P.C. as special insurance counsel iv Legal Analysis Systems, Inc., as Asbestos-Related Bodily Injury Consultant v defunct firm, L. Tersigni Consulting, P.C. as financial advisor, and vi Professor Elizabeth Warren, as a consultant to Caplin  Drysdale, Chartered.  The Asbestos Committee is presently represented by Douglas A. Campbell, Esq., and Philip E. Milch, Esq., at Campbell  Levine, LLC and Peter Van N. Lockwood, Esq., Leslie M. Kelleher, Esq., and Jeffrey A. Liesemer, Esq., at Caplin  Drysdale, Chartered."


#    sentence = svo.pre_process(sentence)
#    sentences =  svo.sentence_split(sentence)
    
       
    
# =============================================================================
#     # method 1: list Subject-Verb-Object structure of the sentences
# =============================================================================
#    val = []
#    for sent in sentences:
#        root_tree = next(svo.get_parse_tree(sent))
#        val.append(svo.process_parse_tree(root_tree))
#
#    print (val)
    
    
    
    
    
# =============================================================================
#     # method 2: split complex sentences to be simple ones and clauses.
# =============================================================================
#    clauses = []
#    for sent in sentences:
#        root_tree = next(svo.get_parse_tree(sent))
#        clauses += svo.retrive_clauses(root_tree)
    
        
#    print("In house method: ")
#    for cl in clauses:
#        doc = nlp(' '.join(cl.leaves()))
#        print(' '.join(cl.leaves()))
#        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
#        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])
    


  
    
# =============================================================================
#     # method 3: retrieve all the subtrees
# =============================================================================
#    clauses = []
#    for t in root_tree.subtrees(filter=(lambda s: s.label()=='S')): clauses.append(t)
     
    
    
# =============================================================================
#     # method 4: stanfard NER to compare results
# =============================================================================
    
#    for sent in sentences:
#        print("Stanford Tagger: ")
#        print(st.tag(sent.split()))
#        print("Spacy Tagger: ")
#        doc = nlp(sent)
#        print('Entities', [(ent.text, ent.label_) for ent in doc.ents if ent.label_=='ORG'])

        
# =============================================================================
#     # method 5: spacy to compare results
# =============================================================================
    
    
    
    

    
    
    
    

