# https://www.kaggle.com/marijakekic/cnn-in-keras-with-pretrained-word2vec-weights
# https://towardsdatascience.com/meme-search-using-pretrained-word2vec-9f8df0a1ade3
# Gensin Word Embeddings
import gensim
import pandas as pd
import numpy as np

word2vec_pre = gensim.models.KeyedVectors.load_word2vec_format('../pretrained word embeddings/GoogleNews-vectors-negative300.bin', binary = True)

# now applything word vector averaging to collections using the same logic for queries
passages = pd.read_table('../data/collection.tsv', names = ['pid', 'text'])
passages['text'] = passages['text'].apply(lambda x: x.split())
passages['text'] = passages['text'].apply(lambda x: [word for word in x if word in word2vec_pre.vocab])
passages = passages[passages['text'].apply(lambda x: len(x)) > 0]
passages['wva'] = passages['text'].apply(lambda x: np.array(sum(word2vec_pre[x])/len(x)))

passages.to_csv('collection_wva.csv')