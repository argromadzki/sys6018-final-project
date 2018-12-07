#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 20:29:12 2018

@author: SM

@description: Read in data files for MSMARCO passage re-ranking
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns

# Read in data

proj_dir = "./"
os.chdir(proj_dir)


qrels_train = pd.read_table("data/qrels.train.tsv")
queries_train = pd.read_table("data/queries.train.tsv")
#passages_train = ps.read_table()


num_lines = sum(1 for l in open("data/top1000.dev.tsv"))

skip_idx = [x for x in range(1, num_lines) if x % 100 != 0]
top1000dev_ids = pd.read_table("data/top1000.dev.tsv", usecols = [0,1])

# descriptive statistics

qrels_train.shape
qrels_train.columns = ['qid', 'pid']
qrels_train.head()
sns.distplot(qrels_train.groupby(by=['qid']).count())

#top1000dev_sample.shape
#top1000dev_sample.columns = ['qid', 'pid', 'query', 'passage']
#top1000dev_sample.head(n=100)

top1000dev_ids.columns = ['qid', 'pid']

sns.distplot(top1000dev_ids.groupby(by=['qid']).count())

top1000dev_ids.groupby(by=['qid']).count()
# 1000 passages per query
len(top1000dev_ids['qid'].unique())
# 6980 distinct queries



