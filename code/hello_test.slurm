#!/bin/bash
#SBATCH --nodes=1             
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00       
#SBATCH --partition=standard    
#SBATCH --account=msg-research-lab
#SBATCH --mem=16000

module purge
module load anaconda3
source activate gensim_env
python3 hello_imports.py

