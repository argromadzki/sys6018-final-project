import os
import numpy as np
import pandas as pd
import sklearn as slr
import scipy as sp
import nltk


def tfidf (file_name):
    
    # Set wd by going up from code and shared repo and into the "data" folder
    os.chdir("..//..//data") # make sure that if you are running this function multiple times in a file to comment this out

    file_table = pd.read_table(file_name+".tsv") # note, don't include tsv as an argument!!

    # clean columns
    file_table.columns = ['qid', 'text']

    # tokenization
    porter = nltk.stem.porter.PorterStemmer()
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    file_table['text'] = file_table['text'].apply(str.lower)
    file_table['tokens']=file_table['text'].apply(nltk.word_tokenize)
    file_table['tokens']=[[porter.stem(x) for x in tokens] for tokens in file_table['tokens']]
    file_table['tokens']=[[w for w in tokens if not w in stop_words] for tokens in file_table['tokens']]
    file_table['tokens']=[[w for w in tokens if w.isalpha()] for tokens in file_table['tokens']]
    file_table['newtext'] = file_table['tokens'].apply(lambda x: " ".join(x))

    file_table.to_csv(file_name+"_tokenized.csv")

    # Create reduced-dimension TDM
    vectorizer_min = slr.feature_extraction.text.CountVectorizer(min_df=.0005)

    file_table_tdm_reduced = vectorizer_min.fit_transform(file_table['newtext'].astype('U'))

    # tfidf
    transformer = slr.feature_extraction.text.TfidfTransformer(smooth_idf=False)

    file_table_tfidf = transformer.fit_transform(file_table_tdm_reduced)
    sp.sparse.save_npz(file_name+"_tfidf.npz", file_table_tfidf)

