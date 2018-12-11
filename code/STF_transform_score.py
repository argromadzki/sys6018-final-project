#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:08:25 2018

@description: Run full read, transform, train, predict 
              This script assumes pre-processed text data
@author: spm9r
"""

# Import modules
import os
import pandas as pd
import sklearn
import sklearn.feature_extraction
import gensim
import logging
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, similarities, models
from gensim.parsing.preprocessing import remove_stopwords

data_dir = data_dir = '/scratch/spm9r'
os.chdir(data_dir)

# Read in all data files

"""
This function takes a parameter that indicates how to read in and preprocess
the data sets. These steps are defined in separate functions.
"""
def get_data_sets(steps):
    if(steps == "usePreProc"):
        return(get_data_usePreProc())

"""
Helper funtion for get_data_steps; data has been stemmed, stripped and 
stopwords removed.
"""
def get_data_usePreProc():
    query_train_text = pd.read_csv("train_query_corpus.csv", index_col=0)
    query_train_text = query_train_text.set_index("qid").iloc[:,2]
    query_train_text = query_train_text[query_train_text.notnull()]
    query_train_text = query_train_text.apply(lambda x: x.split(" "))
        
    query_test_text = pd.read_csv("test_query_corpus.csv", index_col=0)
    query_test_text = query_test_text.set_index("qid").iloc[:,2]
    query_test_text = query_test_text[query_test_text.notnull()]
    query_test_text = query_test_text.apply(lambda x: x.split(" "))
        
    num_lines = sum(1 for l in open("all_passages_corpus.csv"))
    passages_text = pd.read_csv("all_passages_corpus.csv", index_col=0, skiprows=range(10000,num_lines))
    passages_text = passages_text.set_index("pid").iloc[:,0]
    passages_text = passages_text[passages_text.notnull()]
    passages_text = passages_text.apply(lambda x: x.split(" "))
        
    return((query_train_text, query_test_text, passages_text))



# This is what will be the main function
set1, set2, set3 = get_data_sets("usePreProc")

