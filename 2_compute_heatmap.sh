#!/bin/bash -e

#SBATCH --job-name          2_compute_heatmap
#SBATCH --mem               2G
#SBATCH --time              00:05:00
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a
source venv/bin/activate

echo "JOB STARTED"
date

python 2_compute_heatmap.py psp 

echo "FINISHED"
