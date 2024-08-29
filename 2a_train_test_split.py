# Train-test split

import glob
import numpy as np
import os
import shutil
import src.params as params

# Randomly select 20% of the pickle files in data/processed/psp, and move them into a subfolder called 'test'
# The remaining 80% will be moved into a subfolder called 'train'

# Get all the pickle files in data/psp/processed
data_path_prefix = params.data_path_prefix
processed_files = sorted(glob.glob(data_path_prefix + "data/processed/psp/psp_*.pkl"))

train_frac = 0.8

# Randomly shuffle the list of pickle files
np.random.seed(123)  # For reproducibility
np.random.shuffle(processed_files)

# Split the list of pickle files into a training set and a test set
n_files = len(processed_files)
n_train = int(train_frac * n_files)

train_files = processed_files[:n_train]
test_files = processed_files[n_train:]

# Create the 'train' and 'test' subfolders if they don't already exist
os.makedirs(data_path_prefix + "data/processed/psp/train", exist_ok=True)

# Delete any existing files in the 'train' subfolder
#for file in glob.glob(data_path_prefix + "data/processed/psp/train/psp_*.pkl"):
#    os.remove(file)

# Move the training files into the 'train' subfolder
for file in train_files:
    shutil.move(file, file.replace("processed/psp", "processed/psp/train"))

# Create the 'test' subfolder if it doesn't already exist
os.makedirs(data_path_prefix + "data/processed/psp/test", exist_ok=True)

# Delete any existing files in the 'test' subfolder
#for file in glob.glob(data_path_prefix + "data/processed/psp/test/psp_*.pkl"):
#    os.remove(file)

# Move the test files into the 'test' subfolder
for file in test_files:
    shutil.move(file, file.replace("processed/psp", "processed/psp/test"))

# Check the number of files in each subfolder
train_files = sorted(glob.glob(data_path_prefix + "data/processed/psp/train/psp_*.pkl"))
test_files = sorted(glob.glob(data_path_prefix + "data/processed/psp/test/psp_*.pkl"))

print(f"Number of files in PSP 'train' subfolder: {len(train_files)}")
print(f"Number of files in PSP 'test' subfolder: {len(test_files)}")
