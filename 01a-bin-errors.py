# ## PART 1A: CALCULATE ERRORS FOR VECTOR STATS, PER FILE
# # 3_bin_errors.py
# pe = bin_errors(results[sf_2, acf])
# pickle.dump(pe)


import glob
import pickle
import sys

import numpy as np
import pandas as pd

import src.params as params

data_path_prefix = params.data_path_prefix

spacecraft = "psp"

file_index_test = int(sys.argv[1])
n_bins_list = [25]

input_file_list = [
    sorted(
        glob.glob(
            f"{data_path_prefix}data/processed/{spacecraft}/{spacecraft}_*_all_stats.pkl"
        )
    )
][0]

try:
    with open(input_file_list[file_index_test], "rb") as file:
        intervals = pickle.load(file)
        print(
            f"Loaded {len(intervals)} intervals from {input_file_list[file_index_test]}"
        )
except pickle.UnpicklingError:
    print(f"UnpicklingError encountered in file: {file}. Skipping this file.")
except EOFError:
    print(f"EOFError encountered in file: {file}. Skipping this file.")
except Exception as e:
    print(f"An unexpected pe {e} occurred with file: {file}. Skipping this file.")

# Flatten data into a DataFrame
records = []
stat = "sf"
for interval in intervals:
    for lag, lag_n, stat_value in zip(
        interval["lag"], interval["lag_n"], interval[stat]
    ):  # Unpack stat values
        records.append(
            {
                "spacecraft": interval["spacecraft"],
                "start_time": interval["start_time"],
                "interval_id": interval["interval_id"],
                "version": interval["version"],
                "gap_status": interval["gap_status"],
                "tgp": interval["tgp"],
                "lag": lag,
                "lag_n": lag_n,
                stat: stat_value,
            }
        )

# Convert to DataFrame
sf_versions_long = pd.DataFrame(records)

# Verifying SFs by plotting each version for a given interval

# sf_test = sf_versions_long[
#     (sf_versions_long["start_time"] == "2018-11-02 00:00:00")
#     & (sf_versions_long["interval_id"] == 0)
#     & (sf_versions_long["version"] == 0)
# ]

# for gap_status in ["true", "lint", "naive"]:
#     sf = sf_test[sf_test["gap_status"] == gap_status]
#     plt.plot(
#         sf["lag"], sf[stat], label=f"{gap_status}", linestyle="-"
#     )
# plt.legend()
# plt.xscale("log")
# plt.yscale("log")

# Extract rows where gap_status is not "true"
sfs_orig = sf_versions_long[sf_versions_long["gap_status"] == "true"].drop(
    ["gap_status", "tgp"], axis=1
)
sfs_gapped = sf_versions_long[sf_versions_long["gap_status"] != "true"]

# Merge true and gapped dataframes
sfs_wide = pd.merge(
    sfs_orig,
    sfs_gapped,
    how="inner",
    on=["spacecraft", "start_time", "interval_id", "version", "lag"],
    suffixes=("_orig", ""),
)

sfs_wide["sf_pe"] = (sfs_wide["sf"] - sfs_wide["sf_orig"]) / sfs_wide["sf_orig"] * 100

# Need to fix the lag_n values for the "lint" gap_status for then computing gp

# Create a mapping from identifying columns to naive lag_n values
naive_lag_n = sfs_wide[sfs_wide["gap_status"] == "naive"][
    ["spacecraft", "start_time", "interval_id", "version", "lag", "lag_n"]
]
naive_lag_n = naive_lag_n.rename(columns={"lag_n": "naive_lag_n"})

# Merge the naive lag_n values onto the true dataframe
sfs_wide = sfs_wide.merge(
    naive_lag_n,
    on=["spacecraft", "start_time", "interval_id", "version", "lag"],
    how="left",
)

# Now replace the lag_n values where gap_status is "lint"
sfs_wide.loc[sfs_wide["gap_status"] == "lint", "lag_n"] = sfs_wide.loc[
    sfs_wide["gap_status"] == "lint", "naive_lag_n"
]

# Drop the temporary column
sfs_wide = sfs_wide.drop(columns=["naive_lag_n"])

# Get the missing percentage for each lag
sfs_wide["gp"] = sfs_wide["lag_n"] / sfs_wide["lag_n_orig"]


for gap_status in ["lint", "naive"]:
    inputs = sfs_wide[sfs_wide["gap_status"] == gap_status]

    x = inputs["lag"]
    y = inputs["gp"]
    z = inputs["sf"]

    for dim in [2, 3]:
        for n_bins in n_bins_list:
            print(
                f"Grouping sf errors using {gap_status.upper()} into {dim}x{n_bins} bins for {input_file_list[file_index_test]}"
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

                # For every x and y bin, save all the values of sf_pe (not the mean) in those bins to an array
                for i in range(n_bins):
                    for j in range(n_bins):
                        if len(x[(xidx == i) & (yidx == j)]) > 0:
                            pe[i, j] = inputs["sf_pe"][(xidx == i) & (yidx == j)].values

            elif dim == 3:
                if gap_status == "lint":
                    pe = np.full(
                        (n_bins, n_bins, n_bins), dtype=object, fill_value=np.nan
                    )

                    for i in range(n_bins):
                        for j in range(n_bins):
                            for k in range(n_bins):
                                if len(x[(xidx == i) & (yidx == j) & (zidx == k)]) > 0:
                                    pe[i, j, k] = inputs["sf_pe"][
                                        (xidx == i) & (yidx == j) & (zidx == k)
                                    ].values
            # Condition the following on not 3d and naive
            if dim == 3 and gap_status == "naive":
                pass
            else:
                output_file_path = (
                    input_file_list[file_index_test]
                    .replace("train", "train/errors")
                    .replace(".pkl", f"_pe_{dim}d_{n_bins}_bins_{gap_status}_NEW.pkl")
                )

                with open(
                    output_file_path,
                    "wb",
                ) as f:
                    pickle.dump(pe, f)
                    print(f"Saved binned error array to {output_file_path}")
