#!/bin/bash

#SBATCH --job-name          4b_compute_training_stats
#SBATCH --mem               1G 
#SBATCH --time              00:05:00
#SBATCH --output            logs/%x_%j.out
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a
source venv/bin/activate
# If running (locally) on Windows, may need to change above lines to the following: 
#source venv/Scripts/activate

echo "JOB STARTED"
date
run_mode=mini # "mini" or "full"

python 4b_compute_training_stats.py > results/$run_mode/training_stats.txt

echo "JOB FINISHED"
date
