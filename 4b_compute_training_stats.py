# COMPUTE SOME STATS FOR TRAINING DATA
# Likely can only run this on a subset of the data, due to memory constraints

import src.sf_funcs as sf
import glob
import pandas as pd
import src.params as params
import matplotlib.pyplot as plt
# import matplotlib.cbook

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# Annoying deprecation warning

data_path_prefix = params.data_path_prefix

spacecraft = "psp"
input_file_list = [
    sorted(
        glob.glob(
            f"{data_path_prefix}data/processed/{spacecraft}/train/{spacecraft}_*.pkl"
        )
    )
][0]


(
    files_metadata,
    ints_metadata,
    ints_gapped_metadata,
    sfs,
    sfs_gapped,
) = sf.get_all_metadata(
    input_file_list,
    include_sfs=True,
)  ######## LIMIT N FILES HERE ! ! ! #########

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
    f"\nMemory usage of sfs_gapped (for plotting trendline graphs): {sfs_gapped.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB\n"
)

for gap_handling in sfs_gapped.gap_handling.unique():
    sf.plot_error_trend_line(
        sfs_gapped[sfs_gapped["gap_handling"] == gap_handling],
        estimator="sf_2",
        title=f"SF estimation error ({gap_handling.upper()}) vs. lag and global sparsity",
        y_axis_log=True,
    )
    plt.savefig(
        f"plots/temp/train_{spacecraft}_error_trend_{gap_handling.upper()}.png",
        bbox_inches="tight",
    )
    plt.close()
