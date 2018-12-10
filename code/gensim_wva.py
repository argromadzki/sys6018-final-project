# https://www.kaggle.com/marijakekic/cnn-in-keras-with-pretrained-word2vec-weights
# https://towardsdatascience.com/meme-search-using-pretrained-word2vec-9f8df0a1ade3
# Gensin Word Embeddings
import gensim
import pandas as pd
import nltk
import numpy as np

word2vec_pre = gensim.models.KeyedVectors.load_word2vec_format('pretrained word embeddings\GoogleNews-vectors-negative300.bin', binary = True)


queries_train = pd.read_table('data\queries.train.tsv', names = ['qid', 'text'])

queries_train['text'] = queries_train['text'].apply(lambda x: x.split())
queries_train['text'] = queries_train['text'].apply(lambda x: [word for word in x if word in word2vec_pre.vocab])

# checking if any queries are empty
(queries_train['text'].apply(lambda x: len(x)) == 0).value_counts()

# deleting these queries
queries_train = queries_train[queries_train['text'].apply(lambda x: len(x)) > 0]

# now applying word vector averaging
queries_train['wva'] = queries_train['text'].apply(lambda x: sum(word2vec_pre[x])/len(x))

# now applything word vector averaging to collections using the same logic for queries
passages = pd.read_table('data\collection.tsv', names = ['pid', 'text'])
passages['text'] = passages['text'].apply(lambda x: x.split())
passages['text'] = passages['text'].apply(lambda x: [word for word in x if word in word2vec_pre.vocab])
passages = queries_train[queries_train['text'].apply(lambda x: len(x)) > 0]
passages['wva'] = passages['text'].apply(lambda x: sum(word2vec_pre[x])/len(x))


#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical