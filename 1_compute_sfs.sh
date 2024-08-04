#!/bin/bash

#SBATCH --job-name          1_compute_sfs
#SBATCH --mem               1G
#SBATCH --array             0-4
#SBATCH --time              00:15:00
#SBATCH --output            logs/%x_%A_%a.out
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

mkdir -p logs/

module load Python/3.10.5-gimkl-2022a
source venv/bin/activate

echo "JOB STARTED"
date

# Set number of files to be processed by each task
#n_files=5 # Adjust this value as needed
task_id=$SLURM_ARRAY_TASK_ID

# Calculate start and end indices for this task
#start_index=$(( task_id * n_files )) 
#end_index=$(( start_index + n_files - 1 ))

# Print the range of file indices for this task

#echo "Task ID: $task_id processing files from $start_index to $end_index"

# Process each file in the range (use $file_index for this)
#for file_index in $(seq $start_index $end_index); do
python 1_compute_sfs.py psp $task_id
#done

echo "JOB FINISHED"
date
