# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:30:00 2018

@author: XuL
"""


#sys.setdefaultencoding("utf-8")
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


class SVO(object):
    
    
    """
    Class Methods to Extract Subject Verb Object Tuples from a Sentence
    """
    def __init__(self):
        """
        Initialize the SVO Methods
        """
        
        self.noun_types = ["NP"]
        self.sent_types = ["S", "SQ", "SBAR", "SBARQ", "SINV", "FRAG"]
    
    
    
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
    
    
    
    
    # find the subject before keywords
    def process_parse_tree(self,parse_tree, keys):
        """
        Returns the Subject  of a Parse Tree.
        Can Vary depending on number of 'sub-sentences' in a Parse Tree
        """
       
        output_list = []
        
    
        for idx, subtree in enumerate(parse_tree[0].subtrees()):
            output_dict ={}
            subject =None
            if subtree.label() in self.sent_types:
                children_list = subtree
                children_values = [each_child.label() for each_child in children_list]
                children_dict = dict(zip(children_values,children_list))
        
                
                # only extract nountype words from sentences contain keywords
                if children_dict.get("NP") is not None:
                    v = children_dict.get("VP")
                    if  v is not None:
                        leaves = v.leaves()
                        leaves = [lemmatizer.lemmatize(leaf,'v') for leaf in leaves]
                        haskey = len([1 for leaf in leaves if leaf in keys])
                        if haskey > 0:
                            subject = self.get_subject(children_dict["NP"])
    
                try:
                    if subject['subject']: 
                        output_dict['subject'] = subject['subject']                 
                        output_list.append(output_dict)
                except Exception:
                        continue
    
        return output_list

    
    
    
    
    # find relevant sentences
    def match_keywords(self, x, keywords, keyverbs):
        result = 0
        for keyword in keywords:
            if not pd.isnull(re.search(re.sub(r"_", " ",keyword), x, re.IGNORECASE)):
                result = 1
        for keyverb in keyverbs:
            if not pd.isnull(re.search(keyverb, x, re.IGNORECASE)):
                result = 1       
        return result
    
    
    
    



    def if_rm(self, x):
        
        ifrm = False
        if x== '':
            ifrm = True
        else:
    
            rm_p = ['debtor', 'debtors', 'Debtor', 'Debtors', \
                 'It', 'it',  'She', 'she', 'He', 'he', 'They', 'they','each', 'Each', \
                 'Company', 'company', 'Companies','companies',\
                 'lenders','lender','Lender','Lenders',\
                 'defendant', 'defendants', 'Defendant', 'Defendants'\
                 'movant','Movant','Movants','movants',\
                 'creditors','Creditors','creditor' ,'Creditor']
            
            rm_a = ['subsidiaries', "Subsidiaries", 'subsidiary', 'Subsidiary', \
                    'affiliates', 'Affiliates', 'affiliate', 'Affiliate']
            
        
            kp_lst = ['INC', 'LLC', 'GP','PC', 'PA','CORP' ]
            
            
            x = x.split('and')
            x = [i.split(" ") for i in x]
            try:
                prefix_1 = sum([rm in x[0] for rm in rm_p]) > 0
                prefix_2 = sum([kp in x[0] for kp in kp_lst])
                appendix_1 = sum([rm in x[1] for rm in rm_a]) > 0
                appendix_2 = sum([kp in x[1] for kp in kp_lst]) 
                if (prefix_1 and prefix_2==0 and appendix_1  and appendix_2==0):
                    ifrm = True
            except:
                prefix = sum([rm in x[0] for rm in rm_p]) > 0
                appendix = sum([rm in x[0] for rm in rm_a]) > 0
                kep = sum([kp in x[0] for kp in kp_lst])
                if (prefix and (kep==0)) or (appendix and (kep==0)):
                    ifrm = True
            
        return ifrm
    
    
    
    def accuracy_score(self, x, y):
       
        x = x.lower()
        x = re.sub('[^A-Za-z0-9 ]+', '', x)
        x = x.split(" ")
        x = [w for w in x if not w == '' ]
        y = y.lower()
        y = re.sub('[^A-Za-z0-9 ]+', '', y)
        y = y.split(" ")
        y = [w for w in y if not w == '' ]
        x.sort()
        y.sort()
        if x==[] and y ==[]:
            return 100
        elif x==[] or y ==[]:
            return 0
        else:
            if len(x) < len(y):
                score = sum([w in y for w in x])/len(x)
            else:
                score = sum([w in x for w in y])/len(y)
            
        return int(score*100)
