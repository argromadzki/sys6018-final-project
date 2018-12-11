#!/bin/bash
#SBATCH --nodes=1             
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00       
#SBATCH --partition=standard    
#SBATCH --account=msg-research-lab
#SBATCH --mem=64000

module purge
module load anaconda3
source activate gensim_env
python3 ~/MS-MARCO-Analysis/code/STF_transform_score.py 'run5' 'train' 'usePreProc' 'tfidf' 20000
python3 ~/MS-MARCO-Analysis/msmarco_eval.py /scratch/spm9r/qrels.train.tsv /scratch/spm9r/tmp/run5/query_scores.csv

