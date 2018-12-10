
import numpy as np
import pandas as pd
import os
os.chdir('C:\\Users\\alxgr\\Documents\\UVA\\DSI\\Fall 2018\\SYS\\Final Project\\data')

df = pd.read_table('queries.train.tsv')

qid = df.iloc[:,0].sample(100, axis=0)

passages = pd.read_table('collection.tsv')
pid = []
for i in range(0,100):
    sampled_ids = passages.iloc[:,0].sample(1000, axis=0)
    # print(pid)
    pid.extend(sampled_ids) # extend keeps this from being a list of lists, basically adding onto a single list 
                            # --> probably terrible time complexity
    print(i) # progress (each ID /100 has 1000 entries in our dataframe)
                # note it will be 99 rather than 100 since zero indexing python

# qid = passages.iloc[:,0].sample(1000, axis=0) # this is how we generate for a single query each ID
len(pid)
both = {'qid': np.repeat(qid,1000), 'pid': pid}
submission = pd.DataFrame(data=both)
submission = submission.reset_index()
submission = submission.iloc[:,1:]

qrels = pd.read_table('qrels.train.tsv')
# qrels = qrels.reset_index()
# qid = qid.reset_index()

qp_rel_pairs = qrels[qrels.iloc[:,0].isin(list(qid))]
qp_rel_pairs = qp_rel_pairs.drop(columns = [qp_rel_pairs.columns[1], qp_rel_pairs.columns[3]])
qp_rel_pairs.columns = ['qid','pid']
qp_rel_pairs = qp_rel_pairs.astype(int)

submission = submission.append(qp_rel_pairs , ignore_index=True)
submission.to_csv('dummy_submission.tsv',sep='\t')







