#!/bin/bash -e

#SBATCH --job-name          3_correct_test_sfs
#SBATCH --mem               2G
#SBATCH --array             0-10
#SBATCH --time              00:05:00
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a
source venv/bin/activate

echo "JOB STARTED"
date

spacecraft = psp
file_index = $SLURM_ARRAY_TASK_ID

python 3_correct_test_sfs.py $spacecraft $file_index

echo "FINISHED"
