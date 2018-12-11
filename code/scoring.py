from sklearn.metrics.pairwise import cosine_similarity
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity, Similarity
import logging
import itertools


# reading in queries_train
queries_train = pd.read_table('data/queries_wva.csv')

# reading in passages
num_lines = sum(1 for l in open("data/collection_wva.csv"))
passages = pd.read_table("data/collection_wva.csv", skiprows=range(100000,num_lines))


# scoring 

# retrieving scores 
queries_df = queries_train
passages_df = queries_train

# getting queries into matrix format
queries_wva_matrix = np.vstack(queries_df['wva'])
passages_wva_matrix = np.vstack(passages_df['wva'])
len(queries_wva_matrix)

scoring = pd.DataFrame(columns = ['qid', 'pid', 'rank'])

for i in range(0, 2):
    cosine_sim = cosine_similarity(queries_wva_matrix[i:i + 1], passages_wva_matrix)
    cosine_sim = cosine_sim.transpose()
    cosine_sim = pd.DataFrame(data = cosine_sim, columns = ['wva'], index = passages_df['qid']) # need to change 'qid' to 'pid'
    cosine_sim = cosine_sim.sort_values(by = 'wva', axis = 0, ascending = False)
    cosine_sim = cosine_sim.head(n = 1000)
    cosine_sim['rank'] = cosine_sim.rank(axis = 0, ascending = False)
    scoring = scoring.append(pd.DataFrame(np.matrix(data = [np.repeat(queries_df['qid'][i], 1000), cosine_sim.index.values, cosine_sim['rank']]).transpose(), columns = ['qid', 'pid', 'rank']))