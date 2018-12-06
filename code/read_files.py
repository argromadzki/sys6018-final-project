#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 20:29:12 2018

@author: SM

@description: Read in data files for MSMARCO passage re-ranking
"""

import pandas as pd
import os

# Read in data

os.chdir("/Users/SM/DSI/classes/fall2018/SYS6018/FinalProject/")

qrels_train = pd.read_table("data/qrels.train.tsv")
queries_train = pd.read_table("data/queries.train.tsv")


num_lines = sum(1 for l in open("data/top1000.dev.tsv"))
skip_idx = [x for x in range(1, num_lines) if x % 100 != 0]
top1000dev_sample = pd.read_table("data/top1000.dev.tsv", skiprows=skip_idx)

# 