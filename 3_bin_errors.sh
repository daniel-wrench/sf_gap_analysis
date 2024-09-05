#!/bin/bash

#SBATCH --job-name          3_bin_errors
#SBATCH --mem               1G
#SBATCH --array             0-169
#SBATCH --time              03:00:00
#SBATCH --output            logs/%x_%A_%3a.out
##SBATCH --mail-type         BEGIN,END,FAIL
##SBATCH --mail-user         daniel.wrench@vuw.ac.nz

mkdir -p logs/

module load Python/3.10.5-gimkl-2022a
source venv/bin/activate
# If running (locally) on Windows, may need to change above lines to the following: 
#source venv/Scripts/activate

echo "JOB STARTED"
date

dim=3
echo "DIM: $dim"
n_bins=10
echo "N_BINS: $n_bins"

# Specify total number of files
total_files=3

# Set number of files to be processed by each task
n_files=3 # Adjust this value as needed (should really be defined based on number of job array tasks)
task_id=$SLURM_ARRAY_TASK_ID

# Calculate start index for this task (need to set to 0 if running on a single node)
start_index=0

# Calculate the stride (number of files to skip between reads)
stride=$(( total_files / n_files ))

echo "Task ID: $task_id processing every $stride th file, starting from $start_index"

# Process each file based on the stride
for ((file_index=$start_index; file_index < total_files; file_index+=$stride)); do
  echo Core $task_id about to read file $file_index
  python 3_bin_errors.py $file_index $dim $n_bins
done

echo "JOB FINISHED"
date
