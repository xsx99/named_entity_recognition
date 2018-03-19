# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:28:12 2018

@author: XuL
"""

import random
from pathlib import Path
import spacy
import en_core_web_md as spacyEn



TRAIN_DATA = [
    ('Choudries Inc. Super Seven Food Mart filed for Chapter 11 bankruptcy protection Bankr. M.D. Pa. Case No. 16-02475 on June 13, 2016.', {
        'entities': [(0, 35, 'ORG')]
    }),
    ('2747 Camelback,  LLC Test, based in Dallas, Texas, filed a Chapter 11 petition Bankr. N.D. Tex. Case No. 16-31846 on May 4, 2016.', {
        'entities': [(0, 23, 'ORG')]
    })
]




n_iter = 100


"""Load the model, set up the pipeline and train the entity recognizer."""
nlp =  spacyEn.load()
  # create blank Language class
print("Created en_core_web_md model")

# create the built-in pipeline components and add them to the pipeline
# nlp.create_pipe works for built-ins that are registered with spaCy
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
# otherwise, get it so we can add labels
else:
    ner = nlp.get_pipe('ner')

# add labels
for _, annotations in TRAIN_DATA:
    print(annotations)
    for ent in annotations.get('entities'):
        print(ent[2])
        ner.add_label(ent[2])

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            nlp.update(
                [text],  # batch of texts
                [annotations],  # batch of annotations
                drop=0.5,  # dropout - make it harder to memorise data
                sgd=optimizer,  # callable to update weights
                losses=losses)
        print(losses)

# test the trained model
for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


#sentences = []
#sentences.append("")
#sentences.append("")
#sentences.append("")
#sentences.append("")


    
    
    
  






























































