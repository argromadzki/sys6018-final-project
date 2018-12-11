#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:08:25 2018

@description: Run full read, transform, train, predict 
              This script assumes pre-processed text data
@author: spm9r
"""

#runname = "temp" ###FIX parameterize this
runname = sys.argv[0]
#traintest = "test"
traintest = sys.argv[1]
###FIX parameterize whether to use query train or query test

# Import modules
import os
import sys
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
def get_data_sets(steps, traintest):
    if(steps == "usePreProc"):
        return(get_data_usePreProc(traintest))

"""
Helper funtion for get_data_steps; data has been stemmed, stripped and 
stopwords removed.
"""
def get_data_usePreProc(traintest):
    if(traintest == "train"):
        num_lines = sum(1 for l in open("train_query_corpus"))
        query_text = pd.read_csv("train_query_corpus.csv", index_col=0,  skiprows=range(10,num_lines))
    elif(traintest == "test"):
        query_text = pd.read_csv("test_query_corpus.csv", index_col=0)
        
    
    query_text = query_text.set_index("qid").iloc[:,2]
    query_text = query_text[query_text.notnull()]
    query_text = query_text.apply(lambda x: x.split(" "))
        
    num_lines = sum(1 for l in open("some_passages.csv"))
    passages_text = pd.read_csv("some_passages.csv", index_col=0, skiprows=range(100000,num_lines))
    passages_text = passages_text.set_index("pid").iloc[:,0]
    passages_text = passages_text[passages_text.notnull()]
    passages_text = passages_text.apply(lambda x: x.split(" "))
        
    return((query_text, passages_text))

def to_transformation(tran):
    if(tran == "tfidf"):
        return(do_tfidf())
        
def do_tfidf():
    # Create TFIDF matrix from passages corpus
    p_tfidf = models.TfidfModel(p_corpus)
    q_tfidf = models.TfidfModel(q_corpus)
    return(q_tfidf, p_tfidf)



# This is what will be the main function
if not os.path.exists(os.path.join("/scratch/spm9r/tmp",runname)):
    os.makedirs(os.path.join("/scratch/spm9r/tmp",runname))

queries, passages = get_data_sets("usePreProc", traintest)

pas_dictionary = corpora.Dictionary(passages)
pas_dictionary.filter_extremes(keep_n = 1001)

pids = pd.Series(passages.index)
qids = pd.Series(queries.index)

# Create passages corpus and serialize to disk
p_corpus = [pas_dictionary.doc2bow(row) for row in passages]
out_name = os.path.join("/scratch/spm9r/tmp",runname,"p_corpus.mm")
corpora.MmCorpus.serialize(out_name, p_corpus)

###FIX May need to remove words from query corpus that aren't in passages dictionary

# Create query_train corpus and serialize to disk
q_corpus = [pas_dictionary.doc2bow(row) for row in queries]
out_name = os.path.join("/scratch/spm9r/tmp",runname,"query_corpus.mm")
corpora.MmCorpus.serialize(out_name, q_corpus)

# Create models from passages and query files
q_model, p_model = to_transformation("tfidf")

# Calculate similarities
index = similarities.MatrixSimilarity(p_model[p_corpus])

"""
This function creates a dataframe for a query and qid with similarity ranked passages
"""
def do_ranking(qid, query):
    scores = pd.Series(index[query])
    scores.index = pids
    scores = scores.sort_values(ascending=False).head(1000)
    scores = scores[scores.values > 0]
    ranks = scores.rank(method='dense')
    scores_df = pd.DataFrame(pd.Series(np.repeat(qid, scores.count())))
    scores_df['pid'] = scores.index
    scores_df['rank'] = ranks.reset_index(drop=True)
    scores_df.columns = ['qid', 'pid', 'rank']
    return(scores_df)

d = {qid: do_ranking(qid, q_corpus[i]) for i,qid in qids.iteritems()}
results = pd.concat(d[k] for k in d.keys()).sort_values(['qid', 'rank'])
out_name = os.path.join("/scratch/spm9r/tmp",runname,"query_scores.csv")
results.to_csv(out_name, header=False, index=False)
