#!/bin/bash
#SBATCH --nodes=2             
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00       
#SBATCH --partition=standard    
#SBATCH --account=msg-research-lab
#SBATCH --mem=64000

module purge
module load anaconda3
source activate gensim_env
python3 gensim_wva_queries_train.py

