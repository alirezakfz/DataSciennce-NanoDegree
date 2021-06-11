# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 21:01:49 2021

@author: alire
"""


import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
from sqlalchemy import create_engine
import joblib
import gzip

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens




import gzip
import pickle


model = pickle.load(open('classifier.pkl', 'rb'))

filename = 'classifier.gzip'
with gzip.open(filename, 'wb') as f:
    pickle.dump(model, f, protocol=0)