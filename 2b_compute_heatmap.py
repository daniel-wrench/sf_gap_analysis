# 2b. FOR ALL INTERVALS IN TRAINING SET: calculate correction

import pickle
import pandas as pd
import numpy as np
import src.sf_funcs as sf
import glob
import src.params as params
import matplotlib.pyplot as plt
import warnings
# import matplotlib.cbook

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# Annoying deprecation warning

# plt.rc("text", usetex=True)
# plt.rc("font", family="serif", serif="Computer Modern", size=16)

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

if rank == 0:
    spacecraft = "psp"
    input_file_list = [
        sorted(
            glob.glob(
                f"{data_path_prefix}data/processed/{spacecraft}/train/{spacecraft}_*.pkl"
            )
        )
    ][0]

    # Randomly shuffle the list of files, to get a better distribution of # intervals within files when testing with
    # just a sample
    np.random.shuffle(input_file_list)

    (
        files_metadata,
        ints_metadata,
        ints,
        ints_gapped_metadata,
        ints_gapped,
        sfs,
        sfs_gapped,
    ) = sf.load_and_concatenate_dataframes(
        input_file_list[:10]
    )  ######## LIMIT HERE ! ! ! #########

    print(
        "Successfully read in and concatenated {} files, starting with {}\nThese comprise a total of {} gapped intervals".format(
            len(files_metadata), input_file_list[0], len(ints_gapped_metadata)
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

# Broadcast the data to all ranks, if MPI is available
if mpi_available:
    sfs_gapped = comm.bcast(sfs_gapped, root=0)

# Compute and export correction factors, plot as heatmaps

n_bins_list = params.n_bins_list

for n_bins in n_bins_list:
    for gap_handling in ["naive", "lint"]:
        for dim in [2, 3]:
            print(
                "Process {} calculating {}D heatmap for {} with {} bins".format(
                    rank, dim, gap_handling, n_bins
                )
            )

            if dim == 2:
                (
                    xedges,
                    yedges,
                    pe_mean,
                    pe_std,
                    pe_min,
                    pe_max,
                    n,
                    scaling,
                    scaling_lower,
                    scaling_upper,
                ) = sf.get_correction_lookup(
                    sfs_gapped, "missing_percent", dim, comm, gap_handling, n_bins
                )
            elif dim == 3:
                (
                    xedges,
                    yedges,
                    zedges,
                    pe_mean,
                    pe_std,
                    pe_min,
                    pe_max,
                    n,
                    scaling,
                    scaling_lower,
                    scaling_upper,
                ) = sf.get_correction_lookup(
                    sfs_gapped, "missing_percent", dim, comm, gap_handling, n_bins
                )

            if mpi_available:
                # Gather results from all ranks
                all_pe_mean = comm.gather(pe_mean, root=0)
                all_pe_std = comm.gather(pe_std, root=0)
                all_pe_min = comm.gather(pe_min, root=0)
                all_pe_max = comm.gather(pe_max, root=0)
                all_n = comm.gather(n, root=0)
                all_scaling = comm.gather(scaling, root=0)
                all_scaling_lower = comm.gather(scaling_lower, root=0)
                all_scaling_upper = comm.gather(scaling_upper, root=0)

            if rank == 0:
                # I expect to see RuntimeWarnings in this block
                if mpi_available:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        pe_mean = np.nanmean(all_pe_mean, axis=0)
                        pe_std = np.nanmean(all_pe_std, axis=0)
                        pe_min = np.nanmean(all_pe_min, axis=0)
                        pe_max = np.nanmean(all_pe_max, axis=0)
                        n = np.nanmean(all_n, axis=0)
                        scaling = np.nanmean(all_scaling, axis=0)
                        scaling_lower = np.nanmean(all_scaling_lower, axis=0)
                        scaling_upper = np.nanmean(all_scaling_upper, axis=0)

                if dim == 2:
                    # Export these arrays in an efficient manner
                    correction_lookup = {
                        "xedges": xedges,
                        "yedges": yedges,
                        "scaling": scaling,
                        "scaling_lower": scaling_lower,
                        "scaling_upper": scaling_upper,
                        "pe_mean": pe_mean,
                        "pe_std": pe_std,
                        "pe_min": pe_min,
                        "pe_max": pe_max,
                        "n": n,
                    }

                if dim == 3:  # has zedges too
                    # Export these arrays in an efficient manner
                    correction_lookup = {
                        "xedges": xedges,
                        "yedges": yedges,
                        "zedges": zedges,
                        "scaling": scaling,
                        "scaling_lower": scaling_lower,
                        "scaling_upper": scaling_upper,
                        "pe_mean": pe_mean,
                        "pe_std": pe_std,
                        "pe_min": pe_min,
                        "pe_max": pe_max,
                        "n": n,
                    }

                # Export the LINT lookup tables as a pickle file
                if gap_handling == "lint":
                    with open(
                        f"data/processed/correction_lookup_{dim}d_{n_bins}_bins.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(correction_lookup, f)

                # Plot a heatmap of the correction lookup table

                if gap_handling == "naive" and dim == 3:
                    # Not interested in 3D heatmaps for this case
                    pass
                else:
                    sf.plot_correction_heatmap(
                        correction_lookup, dim, gap_handling, n_bins
                    )

    # Potential future sample size analysis

    # Get proportion of nans in the flattened array
    # print(np.isnan(heatmap_bin_counts_2d.flatten()).sum() / len(heatmap_bin_counts_2d.flatten())*100, "% of bins in the grid have no corresponding data")

    # Summarise this array in terms of proportion of missing values, minimum value, and median value
    # print(np.nanmedian(heatmap_bin_counts_2d), "is the median count of each bin")

    # For each heatmap, print these stats.
    # Also, for the correction part, note when there is no corresponding bin.

# Other plots of error trends

for gap_handling in sfs_gapped.gap_handling.unique():
    sf.plot_error_trend_line(
        sfs_gapped[sfs_gapped["gap_handling"] == gap_handling],
        estimator="sf_2",
        title=f"SF estimation error ({gap_handling}) vs. lag and global sparsity",
        y_axis_log=True,
    )
    plt.savefig(
        f"plots/temp/train_{spacecraft}_error_trend_{gap_handling}.png",
        bbox_inches="tight",
    )
