print("line 1")

# Imports
import os
import numpy as np
import pandas as pd
import sklearn as slr
import scipy as sp
import nltk
print("imported ok")
nltk.download('stopwords')
print("stopwords ok")
nltk.download('punkt')
print("punkt ok")

# Set wd by going up from code and shared repo and into the "data" folder
os.chdir("/scratch/spm9r")

print("Reading data")

# read in data, ensuring that we have separate sets for queries and passages
train_q = pd.read_table("queries.train.tsv")
all_passages = pd.read_table("collection.tsv")
test_q = pd.read_table("queries.dev.tsv")

print("Test successful")
