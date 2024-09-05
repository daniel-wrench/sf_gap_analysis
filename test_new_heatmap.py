# Read in just the "sfs_gapped" key value from a pickle file
# and print it to stdout

import pickle
import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import src.params as params

# For parallel correction calculation
try:
    from mpi4py import MPI

    mpi_available = True
except ImportError:
    mpi_available = False

    class FakeMPI:
        def __init__(self):
            self.COMM_WORLD = self

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def gather(self, data, root=0):
            return [data]

        def bcast(self, data, root=0):
            return data

    MPI = FakeMPI()

comm = MPI.COMM_WORLD if mpi_available else MPI
rank = comm.Get_rank()
size = comm.Get_size()


data_path_prefix = params.data_path_prefix

spacecraft = "psp"
file_index_test = 0  # int(sys.argv[2])
missing_measure = "missing_percent"
n_bins = 10
gap_handling = "lint"

input_file_list = [
    sorted(
        glob.glob(
            f"{data_path_prefix}data/processed/{spacecraft}/train/{spacecraft}_*.pkl"
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
    print(f"An unexpected error {e} occurred with file: {file}. Skipping this file.")

# Calculate lag-scale errors (sf_2_pe)
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

# Can use np.histogram2d to get the linear bin edges for 2D
xedges = (
    np.logspace(0, np.log10(x.max()), n_bins + 1) - 0.01
)  # so that first lag bin starts just before 1
xedges[-1] = x.max() + 1
yedges = np.linspace(0, 100, n_bins + 1)  # Missing prop
zedges = np.logspace(-2, 1, n_bins + 1)  # ranges from 0.01 to 10


# Split mean calculation across ranks
bins_per_rank = n_bins // size
start_bin = rank * bins_per_rank
end_bin = (rank + 1) * bins_per_rank if rank < size - 1 else n_bins

# Calculate the mean value in each bin
xidx = np.digitize(x, xedges) - 1  # correcting for annoying 1-indexing
yidx = np.digitize(y, yedges) - 1  # as above

errors = np.full(
    (n_bins, n_bins), dtype=object, fill_value=np.nan
)  # Make this an object array to store lists of values
errors_mean = np.full((n_bins, n_bins), fill_value=np.nan)
errors_min = np.full((n_bins, n_bins), fill_value=np.nan)
errors_max = np.full((n_bins, n_bins), fill_value=np.nan)
errors_std = np.full((n_bins, n_bins), fill_value=np.nan)
n = np.full((n_bins, n_bins), fill_value=np.nan)

# scaling = np.full((n_bins, n_bins), fill_value=1)
# scaling_lower = np.full((n_bins, n_bins), fill_value=1)
# scaling_upper = np.full((n_bins, n_bins), fill_value=1)

# For every x and y bin, save all the values of sf_2_pe (not the mean) in those bins to an array
for i in range(n_bins):
    for j in range(n_bins):
        if len(x[(xidx == i) & (yidx == j)]) > 0:
            errors[i, j] = inputs["sf_2_pe"][(xidx == i) & (yidx == j)].values


# SERIAL JOB TO COMBINE ALL RESULTS
arrays = [errors, errors, errors]

for i in range(n_bins):
    for j in range(n_bins):
        all_errors = [array[i, j] for array in arrays]
        if not np.all(np.isnan(all_errors)):
            all_errors = np.concatenate(all_errors)
            errors_mean[i, j] = np.nanmean(all_errors)
            errors_std[i, j] = np.nanstd(all_errors)
            errors_min[i, j] = np.nanmin(all_errors)
            errors_max[i, j] = np.nanmax(all_errors)
            n[i, j] = len(all_errors)

scaling = 1 / (1 + errors_mean / 100)
scaling_lower = 1 / (1 + (errors_mean + 1 * errors_std) / 100)
scaling_upper = 1 / (1 + (errors_mean - 1 * errors_std) / 100)

scaling[np.isnan(scaling)] = 1
scaling_lower[np.isnan(scaling_lower)] = 1
scaling_upper[np.isnan(scaling_upper)] = 1


# Export the LINT lookup tables as a pickle file
# if gap_handling == "lint":
#     with open(
#         f"data/processed/correction_lookup_{dim}d_{n_bins}_bins.pkl",
#         "wb",
#     ) as f:
#         pickle.dump(correction_lookup, f)

# Plot a heatmap of the correction lookup table

# if gap_handling == "naive" and dim == 3:
#     # Not interested in 3D heatmaps for this case
#     pass
# else:
#     sf.plot_correction_heatmap(
#         correction_lookup, dim, gap_handling, n_bins
#     )

fig, ax = plt.subplots(figsize=(7, 5))
plt.grid(False)
plt.pcolormesh(
    xedges,
    yedges,
    errors_mean.T,
    cmap="bwr",
)
plt.grid(False)
plt.colorbar(label="MPE")
plt.clim(-100, 100)
plt.xlabel("Lag ($\\tau$)")
plt.ylabel("Missing percentage")
plt.title(f"Distribution of missing proportion and lag ({gap_handling})", y=1.1)
ax.set_facecolor("black")
ax.set_xscale("log")
plt.show()
