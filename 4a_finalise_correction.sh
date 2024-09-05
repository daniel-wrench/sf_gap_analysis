#!/bin/bash

#SBATCH --job-name          4a_finalise_correction
#SBATCH --mem               30G 
#SBATCH --time              01:00:00
#SBATCH --output            logs/%x_%j.out
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a
source venv/bin/activate
# If running (locally) on Windows, may need to change above lines to the following: 
#source venv/Scripts/activate

dim=3
echo "DIM: $dim"
n_bins=10
echo "N_BINS: $n_bins"


echo "JOB STARTED"
date

python 4a_finalise_correction.py $dim $n_bins

echo "JOB FINISHED"
date
