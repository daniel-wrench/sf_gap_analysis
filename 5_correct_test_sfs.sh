#!/bin/bash -e

#SBATCH --job-name          5_correct_test_sfs
#SBATCH --mem               1G
#SBATCH --array             0-49 #0-5 for wind
#SBATCH --time              00:03:00
#SBATCH --output            logs/%x_%A_%3a.out
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz


module load Python/3.10.5-gimkl-2022a
source venv/bin/activate
# If running (locally) on Windows, may need to change above lines to the following: 
#source venv/Scripts/activate

echo "JOB STARTED"
date

spacecraft=psp
echo "SPACECRAFT: $spacecraft"
file_index=$SLURM_ARRAY_TASK_ID # doesn't work if running locally

python 5_correct_test_sfs.py $spacecraft $file_index

echo "JOB FINISHED"
date
