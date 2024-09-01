#!/bin/bash

#SBATCH --job-name          2b_compute_heatmap
#SBATCH --mem-per-cpu       4GB 
#SBATCH --ntasks	    15
#SBATCH --time              00:10:00
#SBATCH --output            logs/%x_%j.out
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a
source venv/bin/activate
# If running (locally) on Windows, may need to change above lines to the following: 
#source venv/Scripts/activate

echo "JOB STARTED"
date

srun python 2b_compute_heatmap.py # runs on PSP training data 

echo "JOB FINISHED"
date
