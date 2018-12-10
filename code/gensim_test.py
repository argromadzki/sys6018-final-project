# https://www.kaggle.com/marijakekic/cnn-in-keras-with-pretrained-word2vec-weights
# https://towardsdatascience.com/meme-search-using-pretrained-word2vec-9f8df0a1ade3
# Gensin Word Embeddings
import gensim
#import gensim.downloader as api
from gensim import models
import pandas as pd
import nltk
import numpy as np

#api.info()

queries_train = pd.read_table('/scratch/spm9r/queries.train.tsv', names = ['pid', 'text'])
queries_train_text = queries_train['text']
word2vec_pre = gensim.models.KeyedVectors.load_word2vec_format('/scratch/spm9r/pretrained word embeddings/GoogleNews-vectors-negative300.bin', binary = True)

queries_train['text'] = queries_train['text'].apply(lambda x: x.split())
queries_train['text'] = queries_train['text'].apply(lambda x: [word for word in x if word in word2vec_pre.vocab])

# deleting any empty queries after subsetting words
(queries_train['text'].apply(lambda x: len(x)) == 0).value_counts()

# deleting these queries
queries_train = queries_train[queries_train['text'].apply(lambda x: len(x)) > 0]

# now applying word vector averaging
queries_train['wva'] = queries_train['text'].apply(lambda x: sum(word2vec_pre[x])/len(x))

print(queries_train.shape)
print(queries_train.head())
print(queries_train.columns)

#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical
