#!/bin/bash

#SBATCH --job-name          2b_compute_heatmap
#SBATCH --mem               2G
#SBATCH --time              00:10:00
#SBATCH --output            logs/%x_%j.out
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a
source venv/bin/activate

echo "JOB STARTED"
date

python 2b_compute_heatmap.py # runs on PSP training data 

echo "JOB FINISHED"
date
