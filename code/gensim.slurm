#!/bin/bash
#SBATCH --nodes=1             
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --time=02:00:00       
#SBATCH --partition=standard    
#SBATCH --account=msg-research-lab
#SBATCH -o myjobs.%A_%a.out
#SBATCH -e myjobs.%A_%a.err

module purge
module load anaconda3
source activate gensim_env
#echo "echoing"
python3 gensim_test.py

