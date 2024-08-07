#!/bin/bash -e

#SBATCH --job-name          4_compute_test_results
#SBATCH --mem               10G
#SBATCH --time              00:20:00
#SBATCH --output            logs/%x_%j.out
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a
source venv/bin/activate

echo "JOB STARTED"
date

spacecraft=psp
n_bins=20

python 4_compute_test_results.py $spacecraft $n_bins

echo "JOB FINISHED"
date
