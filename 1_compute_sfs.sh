#!/bin/bash

#SBATCH --job-name          1_compute_sfs
#SBATCH --mem               1G
#SBATCH --array             0-169
#SBATCH --time              03:00:00
#SBATCH --output            logs/%x_%A_%3a.out
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

# Add LaTeX to the PATH
export PATH=/usr/bin:$PATH

mkdir -p logs/

module load Python/3.10.5-gimkl-2022a
source venv/bin/activate
# If running (locally) on Windows, may need to change above lines to the following: 
#source venv/Scripts/activate

echo "JOB STARTED"
date

spacecraft=psp

# Specify total number of files
total_files=850

# Set number of files to be processed by each task
n_files=5 # Adjust this value as needed (should really be defined based on number of job array tasks)
task_id=$SLURM_ARRAY_TASK_ID

# Calculate start index for this task (need to set to 0 if running on a single node)
start_index=$task_id

# Calculate the stride (number of files to skip between reads)
stride=$(( total_files / n_files ))

echo "Task ID: $task_id processing every $stride th file, starting from $start_index"

# Process each file based on the stride
for ((file_index=$start_index; file_index < total_files; file_index+=$stride)); do
  echo Core $task_id about to read file $file_index
  python 1_compute_sfs.py $spacecraft $file_index
done

echo "JOB FINISHED"
date
