#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 00:31:05 2018

@author: SM

@description: Do things on MSMARCO

"""

import os
import pandas as pd
import sklearn
import sklearn.feature_extraction
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora

# Load small data

#proj_dir = "/Users/SM/DSI/classes/fall2018/SYS6018/FinalProject/"
data_dir = data_dir = '/scratch/spm9r'
os.chdir(data_dir)

query_train_text = pd.read_csv("train_query_corpus.csv", index_col=0)
query_train_text = query_train_text.set_index("qid").iloc[:,2]
query_train_text = query_train_text[query_train_text.notnull()]
query_train_text = query_train_text.apply(lambda x: x.split(" "))

query_test_text = pd.read_csv("test_query_corpus.csv", index_col=0)
query_test_text = query_test_text.set_index("qid").iloc[:,2]
query_test_text = query_test_text[query_test_text.notnull()]
query_test_text = query_test_text.apply(lambda x: x.split(" "))

# Create matrix in sklearn sparse matrix format
vectorizer_min = sklearn.feature_extraction.text.CountVectorizer(min_df=.0005)

train_q_tdm_reduced = vectorizer_min.fit_transform(query_train_text.astype('U'))
#passages_tdm_reduced = vectorizer_min.fit_transform(all_passages['newtext'].astype('U'))
#test_q_tdm_reduced = vectorizer_min.transform(test_q['newtext'].astype('U'))


train_corpus = gensim.matutils.Sparse2Corpus(train_q_tdm_reduced, documents_columns=False)

feature_names = vectorizer_min.get_feature_names()
dictionary = {
    key: val for key, val in enumerate(feature_names)
}
print(train_corpus)
print(dictionary) 


dictionary_train = corpora.Dictionary(query_train_text)
dictionary_test = corpora.Dictionary(query_test_text)
dictionary_train.save('/scratch/spm9r/querytrain.dict')  # store the dictionary, for future reference
dictionary_test.save('/scratch/spm9r/querytest.dict')  # store the dictionary, for future reference
#print(dictionary)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
print(corpus)