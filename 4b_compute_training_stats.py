# COMPUTE SOME STATS FOR TRAINING DATA
# Likely can only run this on a subset of the data, due to memory constraints

import glob

import pandas as pd

import src.params as params
import src.sf_funcs as sf

# import matplotlib.cbook

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# Annoying deprecation warning

data_path_prefix = params.data_path_prefix
run_mode = params.run_mode
with_sfs = True
n_files = 3  # If above is True, limit the number of files to read in
spacecraft = "psp"
input_file_list = [
    sorted(
        glob.glob(
            f"{data_path_prefix}data/processed/{spacecraft}/train/{spacecraft}_*.pkl"
        )
    )
][0]

# Following is for plotting error trendlines in next script.
if with_sfs is True:
    (
        files_metadata,
        ints_metadata,
        ints_gapped_metadata,
        sfs,
        sfs_gapped,
    ) = sf.get_all_metadata(
        input_file_list[:n_files],  # Limit the number of files read in
        with_sfs=True,
    )

else:
    (
        files_metadata,
        ints_metadata,
        ints_gapped_metadata,
    ) = sf.get_all_metadata(
        input_file_list,
        with_sfs=False,
    )


print(
    "Successfully read in and concatenated {} files, starting with {}\nThese comprise a total of {} clean intervals".format(
        len(files_metadata), input_file_list[0], len(ints_metadata)
    )
)

# Print summary stats of tc
print("\nSummary stats of correlation time, across original files:")
print(files_metadata["tc"].describe())

# Print summary stats of slope
print("\nSummary stats of slope, across original (sub-)intervals:")
print(ints_metadata["slope"].describe())

if with_sfs is True:
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

    # Print the memory usage of the dataframe in MB
    print(
        f"\nMemory usage of sfs_gapped subset (for plotting trendline graphs locally): {sfs_gapped.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB\n"
    )
    # Export the sfs_gapped dataframe to a pickle file
    sfs_gapped.to_pickle(f"results/{run_mode}/train_{spacecraft}_sfs_gapped_SUBSET.pkl")
    print(
        f"Exported this subset to results/{run_mode}/train_{spacecraft}_sfs_gapped_SUBSET.pkl"
    )
