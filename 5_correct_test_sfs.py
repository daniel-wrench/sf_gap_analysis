# STEP 3: FOR EACH INTERVAL IN TEST SET: apply correction
# As with 1_compute_sfs, only working on one file at a time


import pickle
import pandas as pd
import numpy as np
import src.sf_funcs as sf
import glob
import sys
import src.params as params

times_to_gap = params.times_to_gap
pwrl_range = params.pwrl_range
data_path_prefix = params.data_path_prefix
n_bins_list = params.n_bins_list
spacecraft = sys.argv[1]
file_index_test = int(sys.argv[2])

full_output = True

# Importing processed time series and structure functions
if spacecraft == "wind":
    input_file_list = [
        sorted(glob.glob(data_path_prefix + "data/processed/wind/wi_*v05.pkl"))
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

# Unpack the dictionary
files_metadata = data["files_metadata"]
ints_metadata = data["ints_metadata"]
ints = data["ints"]
ints_gapped_metadata = data["ints_gapped_metadata"]
ints_gapped = data["ints_gapped"]
sfs = data["sfs"]
sfs_gapped = data["sfs_gapped"]

print(
    f"Successfully read in {input_file_list[file_index_test]}. This contains {len(ints_metadata)}x{times_to_gap} intervals"
)


for n_bins in n_bins_list:
    # Importing lookup table
    with open(f"data/processed/correction_lookup_2d_{n_bins}_bins.pkl", "rb") as f:
        correction_lookup_2d = pickle.load(f)
    with open(f"data/processed/correction_lookup_3d_{n_bins}_bins.pkl", "rb") as f:
        correction_lookup_3d = pickle.load(f)

    spacecraft = sys.argv[1]  # "psp" or "wind"
    file_index_test = int(sys.argv[2])
    # this simply refers to one of the files in the test files, not the "file_index" variable referring to the original raw file

    # Apply 2D and 3D scaling to test set, report avg errors
    print(
        f"Correcting {len(ints_metadata)} intervals using 2D error heatmap with {n_bins} bins"
    )
    sfs_lint_corrected_2d = sf.compute_scaling(
        sfs_gapped, 2, correction_lookup_2d, n_bins
    )

    print(
        f"Correcting {len(ints_metadata)} intervals using 3D error heatmap with {n_bins} bins"
    )
    sfs_lint_corrected_2d_3d = sf.compute_scaling(
        sfs_lint_corrected_2d, 3, correction_lookup_3d, n_bins
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

    dup_df = ints_gapped_metadata.replace(
        ["naive", "lint"], ["corrected_2d", "corrected_3d"]
    )
    ints_gapped_metadata = pd.concat([ints_gapped_metadata, dup_df])

    for i in files_metadata.file_index.unique():
        for j in range(len(ints_metadata["file_index"] == i)):
            for k in range(times_to_gap):
                for gap_handling in sfs_gapped_corrected.gap_handling.unique():
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

                    ints_gapped_metadata.loc[
                        (ints_gapped_metadata["file_index"] == i)
                        & (ints_gapped_metadata["int_index"] == j)
                        & (ints_gapped_metadata["version"] == k)
                        & (ints_gapped_metadata["gap_handling"] == gap_handling),
                        "slope",
                    ] = slope

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

    # Calculate slope errors
    ints_gapped_metadata = pd.merge(
        ints_gapped_metadata,
        ints_metadata.drop(["int_start", "int_end"], axis=1),
        how="inner",
        on=["file_index", "int_index"],
        suffixes=("", "_orig"),
    )

    # maybe come back to this method of getting true slopes, could be fun

    # # Create a dictionary from df2 with composite keys
    # value2_dict = df2.set_index(['key1', 'key2'])['value2'].to_dict()

    # # Create a composite key in df1 and map the values
    # df1['composite_key'] = list(zip(df1['key1'], df1['key2']))
    # df1['value2'] = df1['composite_key'].map(value2_dict)

    ints_gapped_metadata["slope_pe"] = (
        (ints_gapped_metadata["slope"] - ints_gapped_metadata["slope_orig"])
        / ints_gapped_metadata["slope_orig"]
        * 100
    )
    ints_gapped_metadata["slope_ape"] = np.abs(ints_gapped_metadata["slope_pe"])

    # Export the dataframes in one big pickle file

    if full_output is True:
        output_file_path = input_file_list[file_index_test].replace(
            ".pkl", f"_corrected_{n_bins}_bins_FULL.pkl"
        )
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
        output_file_path = input_file_list[file_index_test].replace(
            ".pkl", f"_corrected_{n_bins}_bins.pkl"
        )
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
