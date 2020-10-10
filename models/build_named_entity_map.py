# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 20:49:34 2020

@author: herborts
"""

import spacy
import pickle
#!python -m spacy download en_core_web_sm

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

database_filepath = "./../data/DisasterResponsePipelineData.db"


X, Y, category_names = load_data(database_filepath)
X = X["message"]

doc = nlp(text)



text_to_entity_map = {}

for text in X:
    doc = nlp(text)
    for entity in doc.ents:
        text_to_entity_map[entity.text] = entity.label_

pickle.dump(text_to_entity_map, open("text_to_entity_map.pck", "wb"))


for text in X:
    for entity in text_to_entity_map.keys():
        if entity in text:
            text += " " + text_to_entity_map[entity]