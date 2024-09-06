#!/bin/bash

#SBATCH --job-name          6_compute_test_stats
#SBATCH --mem               1G
#SBATCH --time              00:02:00
#SBATCH --output            logs/%x_%j.out
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a
source venv/bin/activate
# If running (locally) on Windows, may need to change above lines to the following: 
#source venv/Scripts/activate

echo "JOB STARTED"
date

spacecraft="psp"
echo "SPACECRAFT: $spacecraft"

python 6_compute_test_stats.py $spacecraft

echo "JOB FINISHED"
date
