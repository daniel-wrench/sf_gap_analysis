# Read in just the "sfs_gapped" key value from a pickle file
# and print it to stdout

import pickle
import glob
import numpy as np
import pandas as pd
import sys

import src.params as params


data_path_prefix = params.data_path_prefix

spacecraft = "psp"
gap_handling = "lint"
missing_measure = "missing_percent"

file_index_test = int(sys.argv[1])
n_bins_list = params.n_bins_list

input_file_list = [
    sorted(
        glob.glob(
            f"{data_path_prefix}data/processed/{spacecraft}/train/{spacecraft}_*v02.pkl"
        )
    )
][0]

try:
    with open(input_file_list[file_index_test], "rb") as file:
        data = pickle.load(file)
        sfs_gapped = data["sfs_gapped"]
        sfs = data["sfs"]
        del data
except pickle.UnpicklingError:
    print(f"UnpicklingError encountered in file: {file}. Skipping this file.")
except EOFError:
    print(f"EOFError encountered in file: {file}. Skipping this file.")
except Exception as e:
    print(f"An unexpected pe {e} occurred with file: {file}. Skipping this file.")

# Calculate lag-scale pe (sf_2_pe)
# Join original and copies dataframes and do column operation
sfs_gapped = pd.merge(
    sfs,
    sfs_gapped,
    how="inner",
    on=["file_index", "int_index", "lag"],
    suffixes=("_orig", ""),
)
sfs_gapped["sf_2_pe"] = (
    (sfs_gapped["sf_2"] - sfs_gapped["sf_2_orig"]) / sfs_gapped["sf_2_orig"] * 100
)

inputs = sfs_gapped[sfs_gapped["gap_handling"] == gap_handling]

x = inputs["lag"]
y = inputs[missing_measure]
z = inputs["sf_2"]

for dim in [2,3]:
    for n_bins in n_bins_list:
    
        print(
            f"Grouping sf errors using {gap_handling.upper()} into {dim}x{n_bins} bins for {input_file_list[file_index_test]}"
        )
    
    
        # Can use np.histogram2d to get the linear bin edges for 2D
        max_lag = params.int_length * params.max_lag_prop
        xedges = (
            np.logspace(0, np.log10(max_lag), n_bins + 1) - 0.01
        )  # so that first lag bin starts just before 1
        xedges[-1] = max_lag + 1
        yedges = np.linspace(0, 100, n_bins + 1)  # Missing prop
        zedges = np.logspace(-2, 1, n_bins + 1)  # ranges from 0.01 to 10
        
        # Calculate the mean value in each bin
        xidx = np.digitize(x, xedges) - 1  # correcting for annoying 1-indexing
        yidx = np.digitize(y, yedges) - 1  # as above
        zidx = np.digitize(z, zedges) - 1
        
        if dim == 2:
            pe = np.full((n_bins, n_bins), dtype=object, fill_value=np.nan)
        
            # For every x and y bin, save all the values of sf_2_pe (not the mean) in those bins to an array
            for i in range(n_bins):
                for j in range(n_bins):
                    if len(x[(xidx == i) & (yidx == j)]) > 0:
                        pe[i, j] = inputs["sf_2_pe"][(xidx == i) & (yidx == j)].values
        
        elif dim == 3:
            if gap_handling == "lint":
                pe = np.full((n_bins, n_bins, n_bins), dtype=object, fill_value=np.nan)
        
                for i in range(n_bins):
                    for j in range(n_bins):
                        for k in range(n_bins):
                            if len(x[(xidx == i) & (yidx == j) & (zidx == k)]) > 0:
                                pe[i, j, k] = inputs["sf_2_pe"][
                                    (xidx == i) & (yidx == j) & (zidx == k)
                                ].values
        
        
        output_file_path = (
            input_file_list[file_index_test]
            .replace("train", "train/errors")
            .replace(".pkl", f"_pe_{dim}d_{n_bins}_bins_{gap_handling.upper()}.pkl")
        )
        # Export the pe array in an efficient manner
        with open(
            output_file_path,
            "wb",
        ) as f:
            pickle.dump(pe, f)
            print(f"Saved binned error array to {output_file_path}")
