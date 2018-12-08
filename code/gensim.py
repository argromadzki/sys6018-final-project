#https://www.kaggle.com/marijakekic/cnn-in-keras-with-pretrained-word2vec-weights
# Gensin Word Embeddings
import gensim
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


queries_train = pd.read_table('data\queries.train.tsv', names = ['pid', 'text'])
queries_train_text = queries_train['text']
word2vec_pre = gensim.models.KeyedVectors.load_word2vec_format('pretrained word embeddings\GoogleNews-vectors-negative300.bin', binary = True)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical