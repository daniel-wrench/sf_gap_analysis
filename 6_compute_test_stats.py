# STEP 4: FOR ALL INTERVALS IN TEST SET: get overall test set results

import glob
import pickle

import numpy as np

import src.params as params
import src.sf_funcs as sf

np.random.seed(123)  # For reproducibility

# Import all corrected (test) files
spacecraft = "wind"
n_bins_list = params.n_bins_list
times_to_gap = params.times_to_gap

data_path_prefix = params.data_path_prefix
run_mode = params.run_mode

for n_bins in n_bins_list:
    print(f"Calculating stats for {spacecraft} data with {n_bins} bins")
    if spacecraft == "psp":
        input_file_list = sorted(
            glob.glob(
                data_path_prefix
                + f"results/{run_mode}/corrected_ints/psp_*_corrected_{n_bins}_bins.pkl"
            )
        )
    elif spacecraft == "wind":
        input_file_list = sorted(
            glob.glob(
                data_path_prefix
                + f"results/{run_mode}/corrected_ints/wi_*_corrected_{n_bins}_bins_FULL.pkl"
            )
        )
    else:
        raise ValueError("Spacecraft must be 'psp' or 'wind'")

    (
        files_metadata,
        ints_metadata,
        ints_gapped_metadata,
    ) = sf.get_all_metadata(
        input_file_list
    )  # LIMIT HERE!!

    print(
        f"Successfully read in and concatenated {len(files_metadata)} files, starting with {input_file_list[0]}\n \
            Now calculating statistics of results for the {len(ints_metadata)} (original) x {times_to_gap} intervals contained within"
    )

    # Print summary stats of tc
    print("\nSummary stats of correlation time, across original files:")
    print(files_metadata["tc"].describe())

    # Print summary stats of slope
    print("\nSummary stats of slope (original SFs):")
    print(ints_metadata["slope"].describe())

    print("\nSummary stats of tce (original SFs):")
    print(ints_metadata["tce"].describe())

    print("\nSummary stats of ttu (original SFs):")
    print(ints_metadata["ttu"].describe())

    print("\nNow proceeding to calculate overall test set statistics")

    # Export final overall dataframes, combined from all outputs
    output_file_path = (
        f"results/{run_mode}/test_{spacecraft}_corrected_{n_bins}_bins.pkl"
    )

    # NOT OUTPUTTING COMMENTED DFS DUE TO EXCESSIVE SIZE
    # The uncommented ones are sufficient for getting overall stats
    # We will use the individual corrected files for case study plots
    with open(output_file_path, "wb") as f:
        pickle.dump(
            {
                "files_metadata": files_metadata,
                "ints_metadata": ints_metadata,
                # "ints": ints,
                "ints_gapped_metadata": ints_gapped_metadata,
                # "ints_gapped": ints_gapped,
                # "sfs": sfs,
                # "sfs_gapped_corrected": sfs_gapped_corrected,
            },
            f,
        )

    # Box plots

    correction_stats = ints_gapped_metadata.groupby("gap_handling")[
        [
            "missing_percent_overall",
            "missing_percent_chunks",
            "slope",
            "slope_pe",
            "slope_ape",
            "mpe",
            "mape",
        ]
    ].agg(["count", "mean", "median", "std", "min", "max"])

    correction_corrs = ints_gapped_metadata.groupby("gap_handling")[
        [
            "missing_percent_overall",
            "missing_percent_chunks",
            "slope",
            "slope_pe",
            "slope_ape",
            "mpe",
            "mape",
        ]
    ].corr()

    # Save as csv
    correction_stats.to_csv(
        f"results/{run_mode}/test_{spacecraft}_corrected_{n_bins}_bins_stats.csv"
    )
    correction_corrs.to_csv(
        f"results/{run_mode}/test_{spacecraft}_corrected_{n_bins}_bins_corr_matrix.csv"
    )

    print("Saved correction stats to csv")
