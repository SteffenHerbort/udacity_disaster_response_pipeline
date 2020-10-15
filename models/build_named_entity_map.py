# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 20:49:34 2020

@author: herborts
"""


"""
for text classification, named entities like names (Peter Parker), 
places (Taj Mahal) or cities (Munich) may not be needed explicitly.
Possibly, the category (name, place, city, ...) is more helpful for a 
classifier since it is only important THAT a city is mentioned, not, WHICH.

Since the 'spacy' module did not work well when pickling and transferring
a trained model, this script extracts the relevant entities from the
samples and provides a conversion as a dictionary.

That dictionary can be loaded from "text_to_entity_map.pck" and used when
cleaning the data before training a model.
"""


import spacy
import pickle
from train_classifier import load_data
#!python -m spacy download en_core_web_sm

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

database_filepath = "./../data/DisasterResponsePipelineData.db"


X, Y, category_names = load_data(database_filepath)
X = X["message"]

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