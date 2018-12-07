
# Imports
import os
import numpy as np
import pandas as pd
import sklearn as slr
import scipy as sp
import nltk

# Set wd by going up from code and shared repo and into the "data" folder
os.chdir("..//..//data")


# read in data, ensuring that we have separate sets for queries and passages
train_q = pd.read_csv("train_q.csv")
train_p = pd.read_csv("train_p.csv")
test_q = pd.read_csv("test_q.csv")
test_p = pd.read_csv("test_p.csv")

''' Each Query and Each Passage should be a separate document
    ... Ought to have two seprate corpora -- one for queries and one for passages
'''

porter = nltk.stem.porter.PorterStemmer()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# lowercase, tokenize & stem words in text

# training set corpus creation: query
train_q['text'] = train_q['text'].apply(str.lower)
train_q['tokens']=train_q['text'].apply(nltk.word_tokenize)
train_q['tokens']=[[porter.stem(x) for x in tokens] for tokens in train_q['tokens']]
train_q['tokens']=[[w for w in tokens if not w in stop_words] for tokens in train_q['tokens']]
train_q['tokens']=[[w for w in tokens if w.isalpha()] for tokens in train_q['tokens']]
train_q['newtext'] = train_q['tokens'].apply(lambda x: " ".join(x))

# training set corpus creation: passage
train_p['text'] = train_p['text'].apply(str.lower)
train_p['tokens']=train_p['text'].apply(nltk.word_tokenize)
train_p['tokens']=[[porter.stem(x) for x in tokens] for tokens in train_p['tokens']]
train_p['tokens']=[[w for w in tokens if not w in stop_words] for tokens in train_p['tokens']]
train_p['tokens']=[[w for w in tokens if w.isalpha()] for tokens in train_p['tokens']]
train_p['newtext'] = train_p['tokens'].apply(lambda x: " ".join(x))

# testing set corpus creation: query
test_q['text'] = test_q['text'].apply(str.lower)
test_q['tokens']=test_q['text'].apply(nltk.word_tokenize)
test_q['tokens']=[[porter.stem(x) for x in tokens] for tokens in test_q['tokens']] # triggers a bug in nltk 3.2
test_q['tokens']=[[w for w in tokens if not w in stop_words] for tokens in test_q['tokens']]
test_q['tokens']=[[w for w in tokens if w.isalpha()] for tokens in test_q['tokens']]
test_q['newtext'] = test_q['tokens'].apply(lambda x: " ".join(x))

# testing set corpus creation: passage
test_p['text'] = test_p['text'].apply(str.lower)
test_p['tokens']=test_p['text'].apply(nltk.word_tokenize)
test_p['tokens']=[[porter.stem(x) for x in tokens] for tokens in test_p['tokens']] # triggers a bug in nltk 3.2
test_p['tokens']=[[w for w in tokens if not w in stop_words] for tokens in test_p['tokens']]
test_p['tokens']=[[w for w in tokens if w.isalpha()] for tokens in test_p['tokens']]
test_p['newtext'] = test_p['tokens'].apply(lambda x: " ".join(x))


# write everything to csv's using pd.write_csv('FileName.csv')
train_q.to_csv("train_query_corpus.csv")
train_p.to_csv("train_passage_corpus.csv")

test_q.to_csv("test_query_corpus.csv")
test_p.to_csv("test_passage_corpus.csv")

''' do we need to read in these files or csvs?  
    ... we did before, but i think it was an unnecessary step
'''

# Create reduced-dimension TDM
vectorizer_min = slr.feature_extraction.text.CountVectorizer(min_df=.0005)

train_q_tdm_reduced = vectorizer_min.fit_transform(train_q['newtext'].astype('U'))
train_p_tdm_reduced = vectorizer_min.fit_transform(train_p['newtext'].astype('U'))

test_q_tdm_reduced = vectorizer_min.transform(test_q['newtext'].astype('U'))
test_p_tdm_reduced = vectorizer_min.transform(test_p['newtext'].astype('U'))

# TFIDF transform
transformer = slr.feature_extraction.text.TfidfTransformer(smooth_idf=False)

train_q_tfidf = transformer.fit_transform(train_q_tdm_reduced)
train_p_tfidf = transformer.fit_transform(train_p_tdm_reduced)

test_q_tfidf = transformer.fit_transform(test_q_tdm_reduced)
test_p_tfidf = transformer.fit_transform(test_p_tdm_reduced)

sp.sparse.save_npz("Training_TF_IDF_Q.npz", train_q_tfidf)
sp.sparse.save_npz("Training_TF_IDF_P.npz", train_p_tfidf)

sp.sparse.save_npz("Testing_TF_IDF_Q.npz", test_q_tfidf)
sp.sparse.save_npz("Testing_TF_IDF_P.npz", test_p_tfidf)