# CASE STUDY PLOTS
# Pre-correction case studies

import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator

import src.params as params

plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams.update(
    {
        "font.size": 10,  # Set font size to match LaTeX (e.g., 10pt)
        "axes.labelsize": 10,  # Label size
        "xtick.labelsize": 10,  # X-axis tick size
        "ytick.labelsize": 10,  # Y-axis tick size
        "legend.fontsize": 10,  # Legend font size
        "figure.titlesize": 10,  # Figure title size
        "figure.dpi": 300,  # Higher resolution figure output
    }
)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

np.random.seed(123)  # For reproducibility

times_to_gap = params.times_to_gap

# Import all corrected (test) files
spacecraft = "wind"
n_bins = 25
# times_to_gap = params.times_to_gap # removing as will only be using this file locally

data_path_prefix = params.data_path_prefix
run_mode = params.run_mode
pwrl_range = params.pwrl_range

index = 0
# 2  # For now, just getting first corrected file
# NOTE: THIS IS NOT THE SAME AS FILE INDEX!
# due to train-test split, file indexes could be anything

# Importing one corrected file

print(f"Calculating stats for {spacecraft} data with {n_bins} bins")
if spacecraft == "psp":
    input_file_list = sorted(
        glob.glob(
            data_path_prefix
            + f"data/corrections/{run_mode}/psp_*_corrected_{n_bins}_bins_with_sfs.pkl"
        )
    )
elif spacecraft == "wind":
    input_file_list = sorted(
        glob.glob(
            data_path_prefix
            + f"data/corrections/{run_mode}/wi_*_corrected_{n_bins}_bins_with_sfs.pkl"
        )
    )
else:
    raise ValueError("Spacecraft must be 'psp' or 'wind'")

all_files_metadata = []
all_ints_metadata = []
all_ints = []
all_ints_gapped_metadata = []
all_ints_gapped = []
all_sfs = []
all_sfs_gapped_corrected = []

for file in input_file_list:
    try:
        with open(file, "rb") as f:
            data = pickle.load(f)
    except pickle.UnpicklingError:
        print(f"UnpicklingError encountered in file: {file}.")
        continue
    except EOFError:
        print(f"EOFError encountered in file: {file}.")
        continue
    except Exception as e:
        print(f"An unexpected error {e} occurred with file: {file}.")
        continue
    print(f"Loaded {file}")

    # Unpack the dictionary and append to lists
    all_files_metadata.append(data["files_metadata"])
    all_ints_metadata.append(data["ints_metadata"])
    all_ints.append(data["ints"])
    all_ints_gapped_metadata.append(data["ints_gapped_metadata"])
    all_ints_gapped.append(data["ints_gapped"])
    all_sfs.append(data["sfs"])
    all_sfs_gapped_corrected.append(data["sfs_gapped_corrected"])

# Concatenate all dataframes
files_metadata = pd.concat(all_files_metadata, ignore_index=True)
ints_metadata = pd.concat(all_ints_metadata, ignore_index=True)
ints_gapped_metadata = pd.concat(all_ints_gapped_metadata, ignore_index=True)
ints_gapped = pd.concat(all_ints_gapped, ignore_index=True)
sfs = pd.concat(all_sfs, ignore_index=True)
sfs_gapped_corrected = pd.concat(all_sfs_gapped_corrected, ignore_index=True)

# Flatten the list of lists for ints
ints = [item for sublist in all_ints for item in sublist]

print(
    f"Successfully read in {input_file_list[index]}. This contains {len(ints_metadata)}x{times_to_gap} intervals"
)

# Parameters for the case study plots

file_index = files_metadata["file_index"].values[0]
# THERE IS NOW JUST ONE FILE INDEX TO WORRY ABOUT
# (so could really remove this indexing from code below)

int_index = 1  # Selecting first (possibly only) interval from this file

print(
    "Currently making plots for file index",
    file_index,
    "and interval",
    int_index,
)

fig, ax = plt.subplots(3, 3, figsize=(7, 5), sharex="col")

file_version_pairs = [
    (5, 16, 0, 0),  # (file_index, version, local_int_index, int_index)
    (80, 13, 3, 1),
    (54, 10, 1, 0),  # previously 0, 6, 0
]

annotate_location = [(0.1, 0.1), (0.1, 0.85), (0.1, 0.1)]
mape_location = [
    [(0.05, 0.9), (0.05, 0.8)],
    [(0.3, 0.2), (0.3, 0.1)],
    [(0.05, 0.9), (0.05, 0.8)],
]

# file_version_pairs = [
#     (5, 16, 1),  # (file_index, version, local_int_index)
#     (0, 0, 0),
#     (54, 10, 2),  # previously 0, 6, 0
# ]

for ax_index, (file_index, version, local_int_index, int_index) in enumerate(
    file_version_pairs
):
    # if len(ints) == 1:
    #     ax[ax_index, 0].plot(
    #         ints[0]["Bx"].values, c="grey"
    #     )  # Just plotting one component for simplicity
    # else:
    ax[ax_index, 0].plot(
        ints[local_int_index]["Bx"].values, c="grey", lw=0.8
    )  # Just plotting one component for simplicity
    # Not currently plotting due to indexing issue: need to be able to index
    # on both file_index and int_index
    ax[ax_index, 0].plot(
        ints_gapped.loc[
            (ints_gapped["file_index"] == file_index)
            & (ints_gapped["int_index"] == int_index)
            & (ints_gapped["version"] == version)
            & (ints_gapped["gap_handling"] == "lint"),
            "Bx",  # Just plotting one component for simplicity
        ].values,
        c="black",
        lw=0.8,
    )

    # Put missing_percent_overall in an annotation
    ax[ax_index, 0].annotate(
        f"({ax_index+1}) TGP = {ints_gapped_metadata.loc[(ints_gapped_metadata['file_index']==file_index) & (ints_gapped_metadata['int_index']==int_index) & (ints_gapped_metadata['version']==version) & (ints_gapped_metadata['gap_handling']=='lint'), 'missing_percent_overall'].values[0]:.1f}\%",
        xy=annotate_location[ax_index],
        xycoords="axes fraction",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Plot the SF
    ax[ax_index, 1].plot(
        sfs.loc[
            (sfs["file_index"] == file_index) & (sfs["int_index"] == int_index),
            "lag",
        ],
        sfs.loc[
            (sfs["file_index"] == file_index) & (sfs["int_index"] == int_index),
            "sf_2",
        ],
        c="grey",
        label="True",
        lw=3,
    )

    ax[ax_index, 1].plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "sf_2",
        ],
        c="indianred",
        label="Naive ({:.1f})".format(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
                & (ints_gapped_metadata["gap_handling"] == "naive"),
                "mape",
            ].values[0]
        ),
        lw=1,
    )

    ax[ax_index, 1].plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "lint"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "lint"),
            "sf_2",
        ],
        c="black",
        label="LINT ({:.1f})".format(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
                & (ints_gapped_metadata["gap_handling"] == "lint"),
                "mape",
            ].values[0]
        ),
        lw=1,
    )

    # Plot the sf_2_pe
    ax[ax_index, 2].plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "lint"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "lint"),
            "sf_2_pe",
        ],
        c="black",
        lw=1,
    )
    ax[ax_index, 2].plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "sf_2_pe",
        ],
        c="indianred",
        lw=1,
    )

    # # plot sample size n on right axis
    # ax2 = ax[ax_index, 2].twinx()
    # ax2.plot(
    #     sfs_gapped_corrected.loc[
    #         (sfs_gapped_corrected["file_index"] == file_index)
    #         & (sfs_gapped_corrected["int_index"] == int_index)
    #         & (sfs_gapped_corrected["version"] == version)
    #         & (sfs_gapped_corrected["gap_handling"] == "naive"),
    #         "lag",
    #     ],
    #     sfs_gapped_corrected.loc[
    #         (sfs_gapped_corrected["file_index"] == file_index)
    #         & (sfs_gapped_corrected["int_index"] == int_index)
    #         & (sfs_gapped_corrected["version"] == version)
    #         & (sfs_gapped_corrected["gap_handling"] == "naive"),
    #         "missing_percent",
    #     ],
    #     c="grey",
    #     # linestyle="--",
    #     lw=1,
    # )

    # Label the % missing
    ax[2, 0].set_xlabel("Time", size=10)
    ax[ax_index, 0].set_ylabel("$B_X$ (normalized)", size=10)
    ax[2, 1].set_xlabel("Lag ($\\tau$)", size=10)
    ax[ax_index, 1].set_ylabel("SF", size=10)
    ax[2, 2].set_xlabel("Lag ($\\tau$)", size=10)
    ax[ax_index, 2].set_ylabel("PE (\%)", size=10)
    # ax2.set_ylabel("\% pairs missing", color="grey")
    # ax2.tick_params(axis="y", colors="grey")
    # ax2.set_ylim(0, 100)

    ax[ax_index, 2].axhline(0, c="grey", linestyle="--")
    ax[ax_index, 2].set_ylim(-100, 100)

    ax[ax_index, 1].set_xscale("log")
    ax[ax_index, 1].set_yscale("log")
    ax[ax_index, 2].set_xscale("log")
    ax[ax_index, 1].legend(loc="lower right", fontsize=6)

    ax[ax_index, 1].xaxis.set_major_locator(
        LogLocator(base=10.0, numticks=10)
    )  # For log scale
    ax[ax_index, 2].xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))

    ax[ax_index, 1].grid(
        True,
        which="major",
        axis="both",
        color="gray",
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )
    ax[ax_index, 2].grid(
        True,
        which="major",
        axis="both",
        color="gray",
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )


# Add titles

fig.align_ylabels(ax[:, 0])

plt.subplots_adjust(wspace=0.5, hspace=0.15)
# plt.show()

plt.savefig(
    f"plots/results/{run_mode}/test_{spacecraft}_case_study_gapping.pdf",
    # bbox_inches="tight",
)

# 5e. Corrected case studies

fig, axs = plt.subplots(
    figsize=(7, 2.2), ncols=3, sharey=True, sharex=True, tight_layout=True
)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
axs[0].set_ylabel("SF")


for ax_index, (file_index, version, local_int_index, int_index) in enumerate(
    file_version_pairs
):
    ax = axs[ax_index]

    ax.plot(
        sfs[(sfs["file_index"] == file_index) & (sfs["int_index"] == int_index)]["lag"],
        sfs[(sfs["file_index"] == file_index) & (sfs["int_index"] == int_index)][
            "sf_2"
        ],
        color="grey",
        label="True",
        lw=3,
    )
    ax.plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "sf_2",
        ],
        color="indianred",
        lw=1,
        label="Naive ({:.1f})".format(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
                & (ints_gapped_metadata["gap_handling"] == "naive"),
                "mape",
            ].values[0]
        ),
    )
    ax.plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "lint"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "lint"),
            "sf_2",
        ],
        color="black",
        lw=1,
        label="LINT ({:.1f})".format(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
                & (ints_gapped_metadata["gap_handling"] == "lint"),
                "mape",
            ].values[0]
        ),
    )
    ax.plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_2d"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_2d"),
            "sf_2",
        ],
        color="blue",
        lw=1,
        label="Corrected (2D) ({:.1f})".format(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
                & (ints_gapped_metadata["gap_handling"] == "corrected_2d"),
                "mape",
            ].values[0]
        ),
    )
    ax.plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_3d_smoothed"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_3d_smoothed"),
            "sf_2",
        ],
        color="purple",
        lw=1,
        label="Corrected (3D), smoothed ({:.1f})".format(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
                & (ints_gapped_metadata["gap_handling"] == "corrected_3d_smoothed"),
                "mape",
            ].values[0]
        ),
    )
    ax.plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
            "sf_2",
        ],
        color="#1b9e77",
        lw=1,
        label="Corrected ({:.1f})".format(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
                & (ints_gapped_metadata["gap_handling"] == "corrected_3d"),
                "mape",
            ].values[0]
        ),
    )
    ax.fill_between(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
            "sf_2_lower",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
            "sf_2_upper",
        ],
        color="#1b9e77",
        alpha=0.2,
    )

    missing = ints_gapped_metadata.loc[
        (ints_gapped_metadata["file_index"] == file_index)
        & (ints_gapped_metadata["int_index"] == int_index)
        & (ints_gapped_metadata["version"] == version),
        "missing_percent_overall",
    ].values

    ax.legend(loc="lower right", fontsize=6)
    ax.semilogx()
    ax.semilogy()

    # PLOTTING HEATMAP IN FIRST PANEL

    # Label test intervals with letters
    # alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    ax.annotate(
        f"({ax_index+1}) TGP = {float(missing[0]):.1f}\%",
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0.05, 0.9),
        textcoords="axes fraction",
        transform=ax.transAxes,
        c="black",
        # bbox=dict(
        #     facecolor="lightgrey", edgecolor="white", boxstyle="round", alpha=0.7
        # ),
    )

    ax.set_xlabel("Lag ($\\tau$)")

    print(f"\nStats for interval version {ax_index+1} as plotted:\n")
    print(files_metadata.loc[files_metadata["file_index"] == file_index])
    print(
        ints_gapped_metadata.loc[
            (ints_gapped_metadata["file_index"] == file_index)
            & (ints_gapped_metadata["int_index"] == int_index)
            & (ints_gapped_metadata["version"] == version)
        ][
            [
                "file_index",
                "int_index",
                "version",
                "missing_percent_overall",
                "gap_handling",
                "mape",
                "slope_pe",
            ]
        ]
    )

    # # Plot the slope of the SF
    # slope_corrected = ints_gapped_metadata.loc[
    #     (ints_gapped_metadata["file_index"] == file_index)
    #     & (ints_gapped_metadata["int_index"] == int_index)
    #     & (ints_gapped_metadata["version"] == version)
    #     & (ints_gapped_metadata["gap_handling"] == "corrected_3d"),
    #     "slope",
    # ].values[0]

    # dif.pltpwrl(
    #     pwrl_range[0],
    #     0.1,
    #     pwrl_range[0],
    #     pwrl_range[1],
    #     slope_corrected,
    #     lw=1,
    #     ls="--",
    #     color="#1b9e77",
    #     label=f"Log-log slope: {slope_corrected:.3f}",
    #     ax=ax,
    # )
    # # Plot the slope of the SF
    # slope_naive = ints_gapped_metadata.loc[
    #     (ints_gapped_metadata["file_index"] == file_index)
    #     & (ints_gapped_metadata["int_index"] == int_index)
    #     & (ints_gapped_metadata["version"] == version)
    #     & (ints_gapped_metadata["gap_handling"] == "naive"),
    #     "slope",
    # ].values[0]

    # dif.pltpwrl(
    #     pwrl_range[0],
    #     0.1,
    #     pwrl_range[0],
    #     pwrl_range[1],
    #     slope_naive,
    #     lw=1,
    #     ls="--",
    #     color="red",
    #     label=f"Log-log slope: {slope_corrected:.3f}",
    #     ax=ax,
    # )
    # # Plot the slope of the SF
    # slope = ints_metadata.loc[
    #     (ints_metadata["file_index"] == file_index)
    #     & (ints_metadata["int_index"] == int_index),
    #     "slope",
    # ].values[0]

    # dif.pltpwrl(
    #     pwrl_range[0],
    #     0.1,
    #     pwrl_range[0],
    #     pwrl_range[1],
    #     slope,
    #     lw=2,
    #     ls="--",
    #     color="grey",
    #     label=f"Log-log slope: {slope_corrected:.3f}",
    #     ax=ax,
    # )
# plt.show()
plt.savefig(
    f"plots/results/{run_mode}/test_{spacecraft}_case_study_correcting_{n_bins}_bins.pdf",
    # bbox_inches="tight",
)
