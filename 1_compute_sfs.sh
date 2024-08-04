#!/bin/bash -e

#SBATCH --job-name          1_compute_sfs
#SBATCH --mem               1G
#SBATCH --array             0-10
#SBATCH --time              00:02:00
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load Python/3.10.5-gimkl-2022a
source venv/bin/activate

echo "JOB STARTED"
date

file_index=$SLURM_ARRAY_TASK_ID

python 1_compute_sfs.py psp $file_index

echo "FINISHED"
