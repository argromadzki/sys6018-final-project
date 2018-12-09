# coding: utf-8

# This script reads in the passages token csvs and creates TF*IDF term document matrix

import os
import numpy as np
import pandas as pd
import sklearn as slr
import sklearn.feature_extraction
import scipy as sp

os.chdir("/scratch/spm9r/")
temp = pd.read_csv("passages/passagesac_tokenized.csv", index_col=0)
temp.head()

# read in dataframes to dictionary
l = os.listdir()
l.sort()
d = {name: pd.read_csv(name, index_col=0) for name in l if name.endswith("_tokenized.csv")}
[print(d[k].shape, k) for k in d.keys()]

# concatenate into one long dataframe
all_passages = pd.concat([d[k] for k in d.keys()])


all_passages.shape # should be ~8.8M lines long

# Optionally could write everything to single large csv using pd.write_csv('FileName.csv') (might need to reset index first)

# Create reduced-dimension TDM
vectorizer_min = slr.feature_extraction.text.CountVectorizer(min_df=.0005)

#train_q_tdm_reduced = vectorizer_min.fit_transform(train_q['newtext'].astype('U'))
passages_tdm_reduced = vectorizer_min.fit_transform(all_passages['newtext'].astype('U'))
#test_q_tdm_reduced = vectorizer_min.transform(test_q['newtext'].astype('U'))

# TFIDF transform
transformer = slr.feature_extraction.text.TfidfTransformer(smooth_idf=False)

#train_q_tfidf = transformer.fit_transform(train_q_tdm_reduced)
passages_tfidf = transformer.fit_transform(passages_tdm_reduced)

#test_q_tfidf = transformer.fit_transform(test_q_tdm_reduced)

#sp.sparse.save_npz("Training_TF_IDF_Q.npz", train_q_tfidf)
sp.sparse.save_npz("Passages_TF_IDF.npz", passages_tfidf)
#sp.sparse.save_npz("Testing_TF_IDF_Q.npz", test_q_tfidf)
