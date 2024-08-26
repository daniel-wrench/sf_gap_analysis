import sys
import pandas as pd
import numpy as np
import src.sf_funcs as sf
import glob
import src.params as params
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# import warnings
# import matplotlib.cbook

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# Annoying deprecation warning

# plt.rc("text", usetex=True)
# plt.rc("font", family="serif", serif="Computer Modern", size=16)

data_path_prefix = params.data_path_prefix

spacecraft = "psp"
n_bins = 10
missing_measure = "missing_percent"
num_bins = 25
gap_handling = "lint"

input_file_list = [
    sorted(
        glob.glob(
            f"{data_path_prefix}data/processed/{spacecraft}/train/{spacecraft}_*.pkl"
        )
    )
][0]

# Randomly shuffle the list of files
np.random.shuffle(input_file_list)

(
    files_metadata,
    ints_metadata,
    ints,
    ints_gapped_metadata,
    ints_gapped,
    sfs,
    sfs_gapped,
) = sf.load_and_concatenate_dataframes(input_file_list)

print(
    "Successfully read in and concatenated {} files, starting with {}\nThese comprise a total of {} gapped intervals".format(
        len(files_metadata), input_file_list[0], len(ints_gapped_metadata)
    )
)


# Calculate lag-scale errors (sf_2_pe)
# Join original and copies dataframes and do column operation


ints_gapped_metadata = pd.merge(
    ints_metadata,
    ints_gapped_metadata,
    how="inner",
    on=["file_index", "int_index"],
    suffixes=("_orig", ""),
)


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

inputs = sfs_gapped[sfs_gapped["gap_handling"] == "lint"]

x = inputs["lag"]
y = inputs[missing_measure]

num_bins = 10


# Can use np.histogram2d to get the linear bin edges for 2D
xedges = (
    np.logspace(0, np.log10(x.max()), num_bins + 1) - 0.01
)  # so that first lag bin starts just before 1
xedges[-1] = x.max() + 1
yedges = np.linspace(0, 100, num_bins + 1)  # Missing prop
zedges = np.logspace(-2, 1, num_bins + 1)  # ranges from 0.01 to 10


xidx = np.digitize(x, xedges) - 1  # correcting for annoying 1-indexing
yidx = np.digitize(y, yedges) - 1  # as above

pe_mean = np.full((num_bins, num_bins), fill_value=np.nan)
pe_min = np.full((num_bins, num_bins), fill_value=np.nan)
pe_max = np.full((num_bins, num_bins), fill_value=np.nan)
pe_std = np.full((num_bins, num_bins), fill_value=np.nan)
n = np.full((num_bins, num_bins), fill_value=np.nan)
scaling = np.full((num_bins, num_bins), fill_value=np.nan)
scaling_lower = np.full((num_bins, num_bins), fill_value=np.nan)
scaling_upper = np.full((num_bins, num_bins), fill_value=np.nan)

# upper = np.full((num_bins, num_bins), fill_value=np.nan)
# lower = np.full((num_bins, num_bins), fill_value=np.nan)
for i in range(num_bins):
    for j in range(num_bins):
        # If there are any values, calculate the mean for that bin
        if len(x[(xidx == i) & (yidx == j)]) > 0:
            # means[i, j] = np.mean(y[(xidx == i) & (yidx == j)])
            current_bin_vals = inputs["sf_2_pe"][(xidx == i) & (yidx == j)]

            pe_mean[i, j] = np.nanmean(current_bin_vals)
            pe_std[i, j] = np.nanstd(current_bin_vals)
            pe_min[i, j] = np.nanmin(current_bin_vals)
            pe_max[i, j] = np.nanmax(current_bin_vals)
            n[i, j] = len(current_bin_vals)

            scaling[i, j] = 1 / (1 + pe_mean[i, j] / 100)
            scaling_lower[i, j] = 1 / (1 + (pe_mean[i, j] + 1 * pe_std[i, j]) / 100)
            scaling_upper[i, j] = 1 / (1 + (pe_mean[i, j] - 1 * pe_std[i, j]) / 100)


# Mean percentage error per bin
fig, ax = plt.subplots(figsize=(7, 5))
plt.grid(False)
plt.pcolormesh(
    xedges,
    yedges,
    pe_mean.T,
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


# APPLYING CORRECTION FACTOR
# (on same original interval, to confirm it works as expected)


x = inputs["lag"]
y = inputs[missing_measure]

num_bins = 10


# Can use np.histogram2d to get the linear bin edges for 2D
xedges = (
    np.logspace(0, np.log10(x.max()), num_bins + 1) - 0.01
)  # so that first lag bin starts just before 1
xedges[-1] = x.max() + 1
yedges = np.linspace(0, 100, num_bins + 1)  # Missing prop
zedges = np.logspace(-2, 1, num_bins + 1)  # ranges from 0.01 to 10


xidx = np.digitize(x, xedges) - 1  # correcting for annoying 1-indexing
yidx = np.digitize(y, yedges) - 1  # as above

# Stick with original value if no bins available
inputs["sf_2_corrected_2d"] = inputs["sf_2"].copy()

for i in range(num_bins):
    for j in range(num_bins):
        # If there are any values, calculate the mean for that bin
        if len(x[(xidx == i) & (yidx == j)]) > 0:
            inputs["sf_2_corrected_2d"][(xidx == i) & (yidx == j)] = (
                inputs["sf_2"][(xidx == i) & (yidx == j)] * scaling[i, j]
            )
            inputs["sf_2_corrected_2d_lower"][(xidx == i) & (yidx == j)] = (
                inputs["sf_2"][(xidx == i) & (yidx == j)] * scaling_lower[i, j]
            )
            inputs["sf_2_corrected_2d_upper"][(xidx == i) & (yidx == j)] = (
                inputs["sf_2"][(xidx == i) & (yidx == j)] * scaling_upper[i, j]
            )

# Smoothed version
inputs["sf_2_corrected_2d_smoothed"] = inputs["sf_2_corrected_2d"].rolling(5).mean()
inputs[["sf_2_orig", "sf_2", "sf_2_corrected_2d", "sf_2_corrected_2d_smoothed"]].plot()
plt.show()

# New version works as expected
# But how do we interpolate? Should we?


# Apply 2D and 3D scaling to test set, report avg errors
print(
    f"Correcting {len(ints_metadata)} intervals using 2D error heatmap with {n_bins} bins"
)
sfs_lint_corrected_2d = sf.compute_scaling(inputs, "missing_percent", lookup_table_2d)


plt.plot(sfs_lint_corrected_2d["lag"], sfs_lint_corrected_2d["sf_2"], label="Original")
plt.plot(
    sfs_lint_corrected_2d["lag"],
    sfs_lint_corrected_2d["sf_2_corrected_2d"],
    label="Corrected",
)
plt.semilogx()
plt.semilogy()
plt.legend()
plt.show()
