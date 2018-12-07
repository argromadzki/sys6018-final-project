
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
train_q = pd.read_table("queries.train.tsv")
all_passages = pd.read_table("collection.tsv")
test_q = pd.read_table("queries.dev.tsv")

# clean columns
train_q.columns = ['qid', 'text']
test_q.columns = ['qid', 'text']
all_passages.columns = ['pid', 'text']

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
all_passages['text'] = all_passages['text'].apply(str.lower)
all_passages['tokens']=all_passages['text'].apply(nltk.word_tokenize)
all_passages['tokens']=[[porter.stem(x) for x in tokens] for tokens in all_passages['tokens']]
all_passages['tokens']=[[w for w in tokens if not w in stop_words] for tokens in all_passages['tokens']]
all_passages['tokens']=[[w for w in tokens if w.isalpha()] for tokens in all_passages['tokens']]
all_passages['newtext'] = all_passages['tokens'].apply(lambda x: " ".join(x))

# testing set corpus creation: query
test_q['text'] = test_q['text'].apply(str.lower)
test_q['tokens']=test_q['text'].apply(nltk.word_tokenize)
test_q['tokens']=[[porter.stem(x) for x in tokens] for tokens in test_q['tokens']] # triggers a bug in nltk 3.2
test_q['tokens']=[[w for w in tokens if not w in stop_words] for tokens in test_q['tokens']]
test_q['tokens']=[[w for w in tokens if w.isalpha()] for tokens in test_q['tokens']]
test_q['newtext'] = test_q['tokens'].apply(lambda x: " ".join(x))


# write everything to csv's using pd.write_csv('FileName.csv')

train_q.to_csv("train_query_corpus.csv")
all_passages.to_csv("all_passages_corpus.csv")
test_q.to_csv("test_query_corpus.csv")

''' do we need to read in these files or csvs?  
    ... we did before, but i think it was an unnecessary step
'''

# Create reduced-dimension TDM
vectorizer_min = slr.feature_extraction.text.CountVectorizer(min_df=.0005)

train_q_tdm_reduced = vectorizer_min.fit_transform(train_q['newtext'].astype('U'))
passages_tdm_reduced = vectorizer_min.fit_transform(all_passages['newtext'].astype('U'))
test_q_tdm_reduced = vectorizer_min.transform(test_q['newtext'].astype('U'))

# TFIDF transform
transformer = slr.feature_extraction.text.TfidfTransformer(smooth_idf=False)

train_q_tfidf = transformer.fit_transform(train_q_tdm_reduced)
passages_tfidf = transformer.fit_transform(passages_tdm_reduced)

test_q_tfidf = transformer.fit_transform(test_q_tdm_reduced)

sp.sparse.save_npz("Training_TF_IDF_Q.npz", train_q_tfidf)
sp.sparse.save_npz("Passages_TF_IDF.npz", passages_tfidf)
sp.sparse.save_npz("Testing_TF_IDF_Q.npz", test_q_tfidf)
