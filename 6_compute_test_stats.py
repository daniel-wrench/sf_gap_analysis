# STEP 4: FOR ALL INTERVALS IN TEST SET: get overall test set results

import pickle
import numpy as np
import src.sf_funcs as sf
import glob
import sys
import src.params as params

np.random.seed(123)  # For reproducibility

# Import all corrected (test) files
spacecraft = sys.argv[1]
n_bins_list = params.n_bins_list
times_to_gap = params.times_to_gap

data_path_prefix = params.data_path_prefix

for n_bins in n_bins_list:
    print(f"Calculating stats for {spacecraft} data with {n_bins} bins")
    if spacecraft == "psp":
        input_file_list = sorted(
            glob.glob(
                data_path_prefix
                + f"data/processed/psp/test/psp_*_corrected_{n_bins}_bins.pkl"
            )
        )
    elif spacecraft == "wind":
        input_file_list = sorted(
            glob.glob(
                data_path_prefix + f"data/processed/wind/wi_*_corrected_{n_bins}_bins.pkl"
            )
        )
    else:
        raise ValueError("Spacecraft must be 'psp' or 'wind'")

    (
        files_metadata,
        ints_metadata,
        _,
        ints_gapped_metadata,
        _,
        _,
        _,
    ) = sf.load_and_concatenate_dataframes(input_file_list[:20]) # LIMIT HERE!!

    print(
        f"Successfully read in and concatenated {len(files_metadata)} files, starting with {input_file_list[0]}\n \
        Now calculating statistics of results for the {len(ints_metadata)}x{times_to_gap} intervals contained within"
    )

    # Print summary stats of tc
    print("\nSummary stats of correlation time, across original files:")
    print(files_metadata["tc"].describe())
    
    # Print summary stats of slope
    print("\nSummary stats of slope (original SFs):")
    print(ints_metadata["slope"].describe())

    print("\nNow proceeding to calculate overall test set statistics")

    # Export final overall dataframes, combined from all outputs
    output_file_path = f"data/processed/test_corrected_{spacecraft}_{n_bins}_bins.pkl"

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
    ].agg(["mean", "median", "std", "min", "max"])

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
        f"plots/temp/test_{spacecraft}_correction_stats_{n_bins}_bins.csv"
    )
    correction_corrs.to_csv(
        f"plots/temp/test_{spacecraft}_correction_corrs_{n_bins}_bins.csv"
    )

    print("Saved correction stats to csv")
