# STEP 3: FOR EACH INTERVAL IN TEST SET: apply correction
# As with 1_compute_sfs, only working on one file at a time


import glob
import pickle
import sys

import numpy as np
import pandas as pd

import src.params as params
import src.sf_funcs as sf
import src.utils as utils  # copied directly from Reynolds project, normalize() added

times_to_gap = params.times_to_gap
pwrl_range = params.pwrl_range
data_path_prefix = params.data_path_prefix
run_mode = params.run_mode

spacecraft = "wind"
file_index_test = int(sys.argv[1])
# this simply refers to one of the files in the test files, not the "file_index" variable referring to the original raw file
n_bins = 25

with_sfs = False

# Importing processed time series and structure functions
if spacecraft == "wind":
    input_file_list = [
        sorted(glob.glob(data_path_prefix + "data/processed/wind/test/wi_*v05.pkl"))
    ][0]
elif spacecraft == "psp":
    input_file_list = [
        sorted(glob.glob(data_path_prefix + "data/processed/psp/test/psp_*v02.pkl"))
    ][0]
else:
    raise ValueError("Spacecraft must be 'psp' or 'wind'")

file = input_file_list[file_index_test]
try:
    with open(file, "rb") as f:
        data = pickle.load(f)
except pickle.UnpicklingError:
    print(f"UnpicklingError encountered in file: {file}.")
except EOFError:
    print(f"EOFError encountered in file: {file}.")
except Exception as e:
    print(f"An unexpected error {e} occurred with file: {file}.")

# Unpack the dictionary, fresh each time, esp. for ints_gapped_metadata re-writing
files_metadata = data["files_metadata"]
ints_metadata = data["ints_metadata"]
ints = data["ints"]
ints_gapped_metadata = data["ints_gapped_metadata"]
ints_gapped = data["ints_gapped"]  # Not used here
sfs = data["sfs"]
sfs_gapped = data["sfs_gapped"]

print(
    f"Successfully read in {input_file_list[file_index_test]}. This contains {len(ints_metadata)}x{times_to_gap} intervals"
)
# Importing lookup table
with open(f"results/{run_mode}/correction_lookup_2d_{n_bins}_bins_lint.pkl", "rb") as f:
    correction_lookup_2d = pickle.load(f)
with open(
    f"results/{run_mode}//correction_lookup_3d_{n_bins}_bins_lint.pkl", "rb"
) as f:
    correction_lookup_3d = pickle.load(f)
with open(
    f"results/{run_mode}/correction_lookup_3d_{n_bins}_bins_lint_SMOOTHED.pkl",
    "rb",
) as f:
    correction_lookup_3d_smoothed = pickle.load(f)

# Apply 2D and 3D scaling to test set, report avg errors
print(
    f"Correcting {len(ints_metadata)} intervals using 2D error heatmap with {n_bins} bins"
)
sfs_lint_corrected_2d = sf.compute_scaling(sfs_gapped, 2, correction_lookup_2d, n_bins)

print(
    f"Correcting {len(ints_metadata)} intervals using SMOOTHED 3D error heatmap with {n_bins} bins"
)
sfs_lint_corrected_2d_3d_smoothed = sf.compute_scaling(
    sfs_lint_corrected_2d, 3, correction_lookup_3d_smoothed, n_bins
)

# Rename smoothed columns so not over-ridden when creating non-smoothed versions below
sfs_lint_corrected_2d_3d_smoothed = sfs_lint_corrected_2d_3d_smoothed.rename(
    columns={
        "sf_2_corrected_3d": "sf_2_corrected_3d_smoothed",
        "sf_2_lower_corrected_3d": "sf_2_lower_corrected_3d_smoothed",
        "sf_2_upper_corrected_3d": "sf_2_upper_corrected_3d_smoothed",
    }
)

print(
    f"Correcting {len(ints_metadata)} intervals using 3D error heatmap with {n_bins} bins"
)
sfs_lint_corrected_2d_3d = sf.compute_scaling(
    sfs_lint_corrected_2d_3d_smoothed,
    3,
    correction_lookup_3d,
    n_bins,
)

correction_wide = sfs_lint_corrected_2d_3d[
    [
        "file_index",
        "int_index",
        "version",
        "lag",
        "missing_percent",
        "sf_2_corrected_2d",
        "sf_2_corrected_3d",
        "sf_2_corrected_3d_smoothed",
    ]
]
correction_long = pd.wide_to_long(
    correction_wide,
    ["sf_2"],
    i=["file_index", "int_index", "version", "lag", "missing_percent"],
    j="gap_handling",
    sep="_",
    suffix=r"\w+",
)
correction_bounds_wide = sfs_lint_corrected_2d_3d[
    [
        "file_index",
        "int_index",
        "version",
        "lag",
        "missing_percent",
        "sf_2_lower_corrected_2d",
        "sf_2_lower_corrected_3d",
        "sf_2_upper_corrected_2d",
        "sf_2_upper_corrected_3d",
        "sf_2_lower_corrected_3d_smoothed",
        "sf_2_upper_corrected_3d_smoothed",
    ]
]

correction_bounds_long = pd.wide_to_long(
    correction_bounds_wide,
    ["sf_2_lower", "sf_2_upper"],
    i=["file_index", "int_index", "version", "lag", "missing_percent"],
    j="gap_handling",
    sep="_",
    suffix=r"\w+",
)

corrections_long = pd.merge(
    correction_long,
    correction_bounds_long,
    how="inner",
    on=[
        "file_index",
        "int_index",
        "version",
        "lag",
        "missing_percent",
        "gap_handling",
    ],
).reset_index()

# Adding the corrections, now as a form of "gap_handling", back to the gapped SF dataframe
sfs_gapped_corrected = pd.concat([sfs_gapped, corrections_long])

# Merging the original SFs with the corrected ones to then calculate errors
sfs_gapped_corrected = pd.merge(
    sfs,
    sfs_gapped_corrected,
    how="inner",
    on=["file_index", "int_index", "lag"],
    suffixes=("_orig", ""),
)

# Now we want to be able to calculate the slope and tce for both the gapped and
# original SFs all using the same nested loop.
# However, we have to be a bit hacky with the true SFS, as they do not have a *version*
# (or *gap_handling*) as have not been gapped multiple times. Here we give them artificial versions,
# which will also ensure they exist in the same frequency as the gapped SFs, which should make
# for more accurate plotting later on, I think.

sfs["gap_handling"] = "true"

sfs_true_full = pd.DataFrame()
for i in range(times_to_gap):
    sfs["version"] = i
    sfs_true_full = pd.concat([sfs_true_full, sfs])

sfs_gapped_corrected = pd.concat([sfs_gapped_corrected, sfs_true_full])

# Calculate lag-scale errors (sf_2_pe)
# This is the first time we calculate these errors, for this specific dataset (they were calculated before for the training set)
#
# Previously this didn't work as we had two sf_2_orig columns as the result of merging a dataframe that had already previously been merged. However, this initial merge is no longer taking place, as it is only now that we are calculating any errors *of any sort, including lag-specific ones*, for this particular dataset.

sfs_gapped_corrected["sf_2_pe"] = (
    (sfs_gapped_corrected["sf_2"] - sfs_gapped_corrected["sf_2_orig"])
    / sfs_gapped_corrected["sf_2_orig"]
    * 100
)

# Calculate interval-scale errors
# This is the first time we do this. We do not need these values for the training set, because we only use that for calculating the correction factor, which uses lag-scale errors..

# Adding rows as placeholders for when we correct with 2D and 3D heatmaps and want to calculate errors

# Define new gap_handling values
new_gap_handling = ["corrected_2d", "corrected_3d", "corrected_3d_smoothed"]

# Duplicate existing rows for each new value
dup_df = ints_gapped_metadata.copy()

# Repeat DataFrame for each new type
dup_df = pd.concat([dup_df.assign(gap_handling=gh) for gh in new_gap_handling])

# Append to original DataFrame
ints_gapped_metadata = pd.concat([ints_gapped_metadata, dup_df], ignore_index=True)


for i in files_metadata.file_index.unique():
    for j in range(len(ints_metadata["file_index"] == i)):
        for k in range(times_to_gap):
            for gap_handling in sfs_gapped_corrected.gap_handling.unique():

                if gap_handling != "true":
                    # Save metadata to the gapped metadata df

                    # Calculate MAPE for 2D and 3D corrected SFs

                    ints_gapped_metadata.loc[
                        (ints_gapped_metadata["file_index"] == i)
                        & (ints_gapped_metadata["int_index"] == j)
                        & (ints_gapped_metadata["version"] == k)
                        & (ints_gapped_metadata["gap_handling"] == gap_handling),
                        "mape",
                    ] = np.mean(
                        np.abs(
                            sfs_gapped_corrected.loc[
                                (sfs_gapped_corrected["file_index"] == i)
                                & (sfs_gapped_corrected["int_index"] == j)
                                & (sfs_gapped_corrected["version"] == k)
                                & (
                                    sfs_gapped_corrected["gap_handling"] == gap_handling
                                ),
                                "sf_2_pe",
                            ]
                        )
                    )

                    ints_gapped_metadata.loc[
                        (ints_gapped_metadata["file_index"] == i)
                        & (ints_gapped_metadata["int_index"] == j)
                        & (ints_gapped_metadata["version"] == k)
                        & (ints_gapped_metadata["gap_handling"] == gap_handling),
                        "mpe",
                    ] = np.mean(
                        sfs_gapped_corrected.loc[
                            (sfs_gapped_corrected["file_index"] == i)
                            & (sfs_gapped_corrected["int_index"] == j)
                            & (sfs_gapped_corrected["version"] == k)
                            & (sfs_gapped_corrected["gap_handling"] == gap_handling),
                            "sf_2_pe",
                        ]
                    )

                # Calculate power-law slope for 2D and 3D corrected SFs
                current_int = sfs_gapped_corrected.loc[
                    (sfs_gapped_corrected["file_index"] == i)
                    & (sfs_gapped_corrected["int_index"] == j)
                    & (sfs_gapped_corrected["version"] == k)
                    & (sfs_gapped_corrected["gap_handling"] == gap_handling)
                ]

                if gap_handling == "corrected_3d":
                    # Smoothing the occassionally jumpy correction by first
                    # converting to logarithmically spaced lags
                    indices = np.logspace(
                        0, np.log10(len(current_int) - 1), 100, dtype=int
                    )
                    indices = np.unique(indices)  # Ensure unique indices

                    current_int = current_int.iloc[indices]

                    current_int.sf_2 = utils.SmoothySpec(current_int.sf_2.values, 5)

                # Fit a line to the log-log plot of the structure function over the given range

                slope = np.polyfit(
                    np.log(
                        current_int.loc[
                            (current_int["lag"] >= pwrl_range[0])
                            & (current_int["lag"] <= pwrl_range[1]),
                            "lag",
                        ]
                    ),
                    np.log(
                        current_int.loc[
                            (current_int["lag"] >= pwrl_range[0])
                            & (current_int["lag"] <= pwrl_range[1]),
                            "sf_2",
                        ]
                    ),
                    1,
                )[0]

                # Get ACF from SF
                # var_signal = np.sum(np.var(input, axis=0))
                var_signal = 3
                # will always be this variance as we are using the standardised 3D SF
                acf_from_sf = 1 - (current_int.sf_2 / (2 * var_signal))
                current_int.loc[:, "acf_from_sf"] = acf_from_sf.astype("float32")

                # Calculate correlation scale from acf_from_sf
                tce = utils.compute_outer_scale_exp_trick(
                    current_int["lag"].values,
                    current_int["acf_from_sf"].values,
                    plot=False,
                )
                # plt.show()
                # NB: if plotting, will not work if tce is not found

                ttu, taylor_scale_u_std = utils.compute_taylor_chuychai(
                    current_int["lag"].values,
                    current_int["acf_from_sf"].values,
                    tau_min=params.tau_min,
                    tau_max=params.tau_max,
                )

                if gap_handling != "true":
                    # Save metadata to the gapped metadata df

                    ints_gapped_metadata.loc[
                        (ints_gapped_metadata["file_index"] == i)
                        & (ints_gapped_metadata["int_index"] == j)
                        & (ints_gapped_metadata["version"] == k)
                        & (ints_gapped_metadata["gap_handling"] == gap_handling),
                        "slope",
                    ] = slope

                    ints_gapped_metadata.loc[
                        (ints_gapped_metadata["file_index"] == i)
                        & (ints_gapped_metadata["int_index"] == j)
                        & (ints_gapped_metadata["version"] == k)
                        & (ints_gapped_metadata["gap_handling"] == gap_handling),
                        "tce",
                    ] = tce

                    ints_gapped_metadata.loc[
                        (ints_gapped_metadata["file_index"] == i)
                        & (ints_gapped_metadata["int_index"] == j)
                        & (ints_gapped_metadata["version"] == k)
                        & (ints_gapped_metadata["gap_handling"] == gap_handling),
                        "ttu",
                    ] = ttu

                elif gap_handling == "true":
                    # Save metadata to the original metadata df

                    ints_metadata.loc[
                        (ints_metadata["file_index"] == i)
                        & (ints_metadata["int_index"] == j),
                        "slope",
                    ] = slope

                    ints_metadata.loc[
                        (ints_metadata["file_index"] == i)
                        & (ints_metadata["int_index"] == j),
                        "tce",
                    ] = tce

                    ints_metadata.loc[
                        (ints_metadata["file_index"] == i)
                        & (ints_metadata["int_index"] == j),
                        "ttu",
                    ] = ttu

# Merge "true" values (ints_metadata) with gapped values (ints_gapped_metadata)
# True value column names will get _orig" suffix, we can simply subtract one from the other
# to get the errors
ints_gapped_metadata = pd.merge(
    ints_gapped_metadata,
    ints_metadata.drop(["int_start", "int_end"], axis=1),
    how="inner",
    on=["file_index", "int_index"],
    suffixes=("", "_orig"),
)


def calculate_errors(df, var):
    df[f"{var}_pe"] = (df[var] - df[f"{var}_orig"]) / df[f"{var}_orig"] * 100
    df[f"{var}_ape"] = np.abs(df[f"{var}_pe"])


# Calculate errors for slope and tce
calculate_errors(ints_gapped_metadata, "slope")
calculate_errors(ints_gapped_metadata, "tce")
calculate_errors(ints_gapped_metadata, "ttu")

# Export the dataframes in one big pickle file

if with_sfs is True:
    output_file_path = (
        input_file_list[file_index_test]
        .replace(
            f"data/processed/{spacecraft}/test",
            f"results/{run_mode}/test_sfs_corrected_subset",
        )
        .replace(
            ".pkl",
            f"_corrected_{n_bins}_bins_with_sfs.pkl",
        )
    )
    print("Exporting full output (with SFs) to", output_file_path)

    with open(output_file_path, "wb") as f:
        pickle.dump(
            {
                "files_metadata": files_metadata,
                "ints_metadata": ints_metadata,
                "ints": ints,
                "ints_gapped_metadata": ints_gapped_metadata,
                "ints_gapped": ints_gapped,
                "sfs": sfs,
                "sfs_gapped_corrected": sfs_gapped_corrected,
            },
            f,
        )
else:
    output_file_path = (
        input_file_list[file_index_test]
        .replace(
            "test",
            "test/corrected",
        )
        .replace(
            ".pkl",
            f"_corrected_{n_bins}_bins.pkl",
        )
    )
    print("Exporting truncated output to", output_file_path)
    with open(output_file_path, "wb") as f:
        pickle.dump(
            {
                "files_metadata": files_metadata,
                "ints_metadata": ints_metadata,
                "ints_gapped_metadata": ints_gapped_metadata,
            },
            f,
        )

# CORRECTION CHECKING PLOT
# import matplotlib.pyplot as plt

# new = sfs_lint_corrected_2d_3d
# check_int = new[(new["int_index"] == 2) & (new["version"] == 4)]

# fig, ax = plt.subplots()
# ax.plot(check_int["lag"], check_int["sf_2"], label="Interpolated")
# ax.plot(check_int["lag"], check_int["sf_2_corrected_2d"], label="Corrected 2D")
# ax.plot(check_int["lag"], check_int["sf_2_corrected_3d"], label="Corrected 3D")

# ax.plot(
#     check_int["lag"],
#     check_int["sf_2_corrected_2d_smoothed"],
#     label="Corrected Smoothed 2D",
# )

# # # Fill between lower and upper bounds
# ax.fill_between(
#     check_int["lag"],
#     check_int["sf_2_lower_corrected_2d"],
#     check_int["sf_2_upper_corrected_2d"],
#     alpha=0.5,
#     color="gray",
# )
# ax.set_ylim(0, 7)
# plt.legend()
# plt.show()
