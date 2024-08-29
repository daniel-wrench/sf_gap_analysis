#!/bin/bash -e

#SBATCH --job-name          4_compute_test_results
#SBATCH --mem               5G
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

spacecraft=psp

for n_bins in 15 20; do
python 4_compute_test_results.py $spacecraft $n_bins
done

echo "JOB FINISHED"
date
