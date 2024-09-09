# CASE STUDY PLOTS
# Pre-correction case studies

import pickle
import glob
import numpy as np
import src.sf_funcs as sf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import src.params as params

np.random.seed(123)  # For reproducibility

times_to_gap = params.times_to_gap

plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=16)

# Import all corrected (test) files
spacecraft = "wind"
n_bins = 10
# times_to_gap = params.times_to_gap # removing as will only be using this file locally

data_path_prefix = params.data_path_prefix
output_path = params.output_path

index = 1  # For now, just getting first corrected file
# NOTE: THIS IS NOT THE SAME AS FILE INDEX!
# due to train-test split, file indexes could be anything

# Importing one corrected file

print(f"Calculating stats for {spacecraft} data with {n_bins} bins")
if spacecraft == "psp":
    input_file_list = sorted(
        glob.glob(
            data_path_prefix
            + f"data/corrections/{output_path}/psp_*_corrected_{n_bins}_bins_FULL.pkl"
        )
    )
elif spacecraft == "wind":
    input_file_list = sorted(
        glob.glob(
            data_path_prefix
            + f"data/corrections/{output_path}/wi_*_corrected_{n_bins}_bins_FULL.pkl"
        )
    )
else:
    raise ValueError("Spacecraft must be 'psp' or 'wind'")

file = input_file_list[index]
try:
    with open(file, "rb") as f:
        data = pickle.load(f)
except pickle.UnpicklingError:
    print(f"UnpicklingError encountered in file: {file}.")
except EOFError:
    print(f"EOFError encountered in file: {file}.")
except Exception as e:
    print(f"An unexpected error {e} occurred with file: {file}.")
print(f"Loaded {file}")

# Unpack the dictionary
files_metadata = data["files_metadata"]
ints_metadata = data["ints_metadata"]
ints = data["ints"]
ints_gapped_metadata = data["ints_gapped_metadata"]
ints_gapped = data["ints_gapped"]
sfs = data["sfs"]
sfs_gapped_corrected = data["sfs_gapped_corrected"]

print(
    f"Successfully read in {input_file_list[index]}. This contains {len(ints_metadata)}x{times_to_gap} intervals"
)


# Also do publication-read plots for heatmaps (as least the LINT versions; we haven't output the correction
# lookup for the naive versions, but we do have plots of these on the HPC from step 2b)

# Below is copied directly from 4a_finalise_correction.py
for dim in [2, 3]:
    with open(
        f"data/corrections/{output_path}/correction_lookup_{dim}d_{n_bins}_bins.pkl",
        "rb",
    ) as f:
        correction_lookup = pickle.load(f)
        gap_handling = "lint"

        if dim == 2:
            xedges = correction_lookup["xedges"]
            yedges = correction_lookup["yedges"]
            pe_mean = correction_lookup["pe_mean"]

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
            plt.title(
                f"Distribution of missing proportion and lag ({gap_handling.upper()})",
                y=1.1,
            )
            ax.set_facecolor("black")
            ax.set_xscale("log")
            plt.savefig(
                f"plots/results/{output_path}/train_heatmap_{n_bins}bins_2d_{gap_handling.upper()}.png",
                bbox_inches="tight",
            )
            plt.close()

        elif dim == 3:
            xedges = correction_lookup["xedges"]
            yedges = correction_lookup["yedges"]
            zedges = correction_lookup["zedges"]
            pe_mean = correction_lookup["pe_mean"]

            fig, ax = plt.subplots(
                1, n_bins, figsize=(n_bins * 3, 3.5), tight_layout=True
            )
            # Remove spacing between subplots
            plt.subplots_adjust(wspace=0.2)
            plt.grid(False)
            for i in range(n_bins):
                ax[i].grid(False)
                c = ax[i].pcolormesh(
                    xedges,
                    yedges,
                    pe_mean[:, :, i],
                    cmap="bwr",
                )
                # plt.colorbar(label="MPE")
                c.set_clim(-100, 100)
                plt.xlabel("Lag ($\\tau$)")
                plt.ylabel("Missing proportion")
                plt.title("Distribution of missing proportion and lag")
                ax[i].set_facecolor("black")
                ax[i].semilogx()
                ax[i].set_title(
                    f"Power bin {i+1}/{n_bins}".format(np.round(zedges[i], 2))
                )
                ax[i].set_xlabel("Lag ($\\tau$)")
                # Remove y-axis labels for all but the first plot
                if i > 0:
                    ax[i].set_yticklabels([])
                    ax[i].set_ylabel("")

            plt.savefig(
                f"plots/results/{output_path}/train_heatmap_{n_bins}bins_3d_{gap_handling.upper()}_power.png",
                bbox_inches="tight",
            )
            plt.close()

            fig, ax = plt.subplots(
                1, n_bins, figsize=(n_bins * 3, 3.5), tight_layout=True
            )
            # Remove spacing between subplots
            plt.grid(False)
            plt.subplots_adjust(wspace=0.2)
            for i in range(n_bins):
                ax[i].grid(False)
                c = ax[i].pcolormesh(
                    yedges,
                    zedges,
                    pe_mean[i, :, :],
                    cmap="bwr",
                )
                # plt.colorbar(label="MPE")
                c.set_clim(-100, 100)
                ax[i].set_xlabel("Missing prop")
                ax[i].set_ylabel("Power")
                plt.title("Distribution of missing proportion and lag")
                ax[i].set_facecolor("black")
                ax[i].semilogy()
                ax[i].set_title(
                    f"Lag bin {i+1}/{n_bins}".format(np.round(zedges[i], 2))
                )
                ax[i].set_xlabel("Missing prop")
                # Remove y-axis labels for all but the first plot
                if i > 0:
                    ax[i].set_yticklabels([])
                    ax[i].set_ylabel("")

            plt.savefig(
                f"plots/results/{output_path}/train_heatmap_{n_bins}bins_3d_{gap_handling.upper()}_lag.png",
                bbox_inches="tight",
            )
            plt.close()

            fig, ax = plt.subplots(
                1, n_bins, figsize=(n_bins * 3, 3.5), tight_layout=True
            )
            # Remove spacing between subplots
            plt.grid(False)
            plt.subplots_adjust(wspace=0.2)
            for i in range(n_bins):
                ax[i].grid(False)
                c = ax[i].pcolormesh(
                    xedges,
                    zedges,
                    pe_mean[:, i, :],
                    cmap="bwr",
                )
                # plt.colorbar(label="MPE")
                c.set_clim(-100, 100)
                plt.title("Distribution of missing proportion and lag")
                ax[i].set_facecolor("black")
                ax[i].semilogx()
                ax[i].semilogy()
                ax[i].set_title(
                    f"Missing prop bin {i+1}/{n_bins}".format(np.round(zedges[i], 2))
                )
                ax[i].set_xlabel("Lag ($\\tau$)")
                ax[i].set_ylabel("Power")
                # Remove y-axis labels for all but the first plot
                if i > 0:
                    ax[i].set_yticklabels([])
                    ax[i].set_ylabel("")

            plt.savefig(
                f"plots/results/{output_path}/train_heatmap_{n_bins}bins_3d_{gap_handling.upper()}_missing.png",
                bbox_inches="tight",
            )
            plt.close()


# Load just the 2D Lint version for use in later case study plots
# USING 15 BINS SO AS TO HIDE EMPTY BINS FROM PLOT
with open(
    f"data/corrections/{output_path}/correction_lookup_2d_15_bins.pkl", "rb"
) as f:
    correction_lookup = pickle.load(f)
    xedges = correction_lookup["xedges"]
    yedges = correction_lookup["yedges"]
    pe_mean = correction_lookup["pe_mean"]


# Count of pairs per bin (same for both)
# fig, ax = plt.subplots(figsize=(7, 5))
# plt.grid(False)
# plt.pcolormesh(
#     heatmap_bin_edges_2d[0],
#     heatmap_bin_edges_2d[1],
#     heatmap_bin_counts_2d.T,
#     cmap="Greens",
# )
# # Remove gridlines
# plt.grid(False)
# plt.colorbar(label="Count of intervals")
# plt.xlabel("Lag ($\\tau$)")
# plt.ylabel("Missing percentage")
# plt.title("Distribution of missing proportion and lag", y=1.1)
# ax.set_facecolor("black")
# ax.set_xscale("log")
# plt.savefig(
#     f"plots/final/train_{spacecraft}_heatmap_{n_bins}bins_2d_counts.png",
#     bbox_inches="tight",
# )


# Parameters for the case study plots

file_index = files_metadata["file_index"].values[0]
# THERE IS NOW JUST ONE FILE INDEX TO WORRY ABOUT
# (so could really remove this indexing from code below)

int_index = 0  # Selecting first (possibly only) interval from this file

print(
    "Currenting making plots for file index",
    file_index,
    "and interval",
    int_index,
)

fig, ax = plt.subplots(2, 3, figsize=(15, 2 * 3))
# will use consistent interval index, but choose random versions of it to plot
versions_to_plot = [
    0,
    1,
    # 23,  # 13 = 9% missing, 23 = 50% missing
    # 8,
]
for ax_index, version in enumerate(versions_to_plot):
    if len(ints) == 1:
        ax[ax_index, 0].plot(
            ints[0]["Bx"].values, c="grey"
        )  # Just plotting one component for simplicity
    else:
        ax[ax_index, 0].plot(
            ints[0][int_index][0]["Bx"].values, c="grey"
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
    )

    # Put missing_percent_overall in the title
    ax[ax_index, 0].set_title(
        f"{ints_gapped_metadata.loc[(ints_gapped_metadata['file_index']==file_index) & (ints_gapped_metadata['int_index']==int_index) & (ints_gapped_metadata['version']==version) & (ints_gapped_metadata['gap_handling']=='lint'), 'missing_percent_overall'].values[0]:.1f}\% missing"
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
        lw=4,
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
    )

    # plot sample size n on right axis
    ax2 = ax[ax_index, 2].twinx()
    ax2.plot(
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
            "missing_percent",
        ],
        c="grey",
        # linestyle="--",
        lw=1,
    )

    # Label the axes
    ax[1, 0].set_xlabel("Time")
    ax[ax_index, 0].set_ylabel("$B_R$ (normalised)")
    ax[1, 1].set_xlabel("Lag ($\\tau$)")
    ax[ax_index, 1].set_ylabel("SF")
    ax[1, 2].set_xlabel("Lag ($\\tau$)")
    ax[ax_index, 2].set_ylabel("\% error")
    ax2.set_ylabel("\% pairs missing", color="grey")
    ax2.tick_params(axis="y", colors="grey")
    ax2.set_ylim(0, 100)

    ax[ax_index, 2].axhline(0, c="black", linestyle="--")
    ax[ax_index, 2].set_ylim(-100, 100)

    ax[ax_index, 1].set_xscale("log")
    ax[ax_index, 1].set_yscale("log")
    ax[ax_index, 2].set_xscale("log")
    ax[ax_index, 1].legend(fontsize=12)
    [ax[0, i].set_xticklabels([]) for i in range(3)]

# Add titles
ax[0, 1].set_title("Structure function estimates")
ax[0, 2].set_title("SF \% error and \% pairs missing")
plt.subplots_adjust(wspace=0.4)

plt.savefig(
    f"plots/results/{output_path}/test_{spacecraft}_case_study_gapping_{file_index}_{int_index}.png",
    bbox_inches="tight",
)

# 5e. Corrected case studies


# Annotate each heatmap trace with info
def annotate_curve(ax, x, y, text, offset_scaling=(0.3, 0.1)):
    # Find the index of y value closest to the median value
    idx = np.argmin(np.abs(y - np.percentile(y, 20)))

    # Coordinates of the point of maximum y value
    x_max = x.iloc[idx]
    y_max = y.iloc[idx]

    # Convert offset from axes fraction to data coordinates
    x_text = 10 ** (offset_scaling[0] * np.log10(x_max))  # Log-axis
    y_text = y_max + offset_scaling[1] * (ax.get_ylim()[1] - ax.get_ylim()[0])

    # Annotate with the text, adjusting the position with xytext_offset
    ax.annotate(
        text,
        xy=(x_max, y_max - 1),
        xytext=(x_text, y_text),
        # xycoords="axes fraction",
        # textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="gray"),
        bbox=dict(facecolor="gray", edgecolor="gray", boxstyle="round", alpha=0.7),
        fontsize=20,
    )


fig = plt.figure(figsize=(13, 4))

# Create a GridSpec layout with specified width ratios and horizontal space
gs1 = GridSpec(1, 1, left=0.06, right=0.35)
gs2 = GridSpec(1, 2, left=0.43, right=0.99, wspace=0)

# Create subplots
ax0 = fig.add_subplot(gs1[0, 0])
ax1 = fig.add_subplot(gs2[0, 0])

for ax_index, version in enumerate(versions_to_plot):
    if ax_index == 0:
        ax = ax1
        ax.set_ylabel("SF")
    else:
        ax = fig.add_subplot(gs2[0, ax_index], sharey=ax1)
        plt.setp(ax.get_yticklabels(), visible=False)

    ax.plot(
        sfs[(sfs["file_index"] == file_index) & (sfs["int_index"] == int_index)]["lag"],
        sfs[(sfs["file_index"] == file_index) & (sfs["int_index"] == int_index)][
            "sf_2"
        ],
        color="black",
        label="True",
        lw=4,
        alpha=0.5,
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
        color="red",
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
    # ax.plot(
    #     sfs_gapped_corrected.loc[
    #         (sfs_gapped_corrected["file_index"] == file_index)
    #         & (sfs_gapped_corrected["int_index"] == int_index)
    #         & (sfs_gapped_corrected["version"] == version)
    #         & (sfs_gapped_corrected["gap_handling"] == "corrected_2d"),
    #         "lag",
    #     ],
    #     sfs_gapped_corrected.loc[
    #         (sfs_gapped_corrected["file_index"] == file_index)
    #         & (sfs_gapped_corrected["int_index"] == int_index)
    #         & (sfs_gapped_corrected["version"] == version)
    #         & (sfs_gapped_corrected["gap_handling"] == "corrected_2d"),
    #         "sf_2",
    #     ],
    #     color="blue",
    #     lw=1,
    #     label="Corrected (2D) ({:.1f})".format(
    #         ints_gapped_metadata.loc[
    #             (ints_gapped_metadata["file_index"] == file_index)
    #             & (ints_gapped_metadata["int_index"] == int_index)
    #             & (ints_gapped_metadata["version"] == version)
    #             & (ints_gapped_metadata["gap_handling"] == "corrected_2d"),
    #             "mape",
    #         ].values[0]
    #     ),
    # )
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

    ax.legend(loc="lower right", fontsize=16)
    ax.semilogx()
    ax.semilogy()

    # PLOTTING HEATMAP IN FIRST PANEL

    c = ax0.pcolormesh(
        xedges,
        yedges,  # convert to \% Missing
        pe_mean.T,
        cmap="bwr",
    )
    # fig.colorbar(c, ax=ax0, label="MPE")
    c.set_clim(-100, 100)
    c.set_facecolor("black")
    ax0.set_xlabel("Lag ($\\tau$)")
    ax0.plot(
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
            "missing_percent",
        ],
        c="grey",
        # linestyle="--",
        lw=1,
    )

    # Label test intervals with letters
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    annotate_curve(
        ax0,
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
            "missing_percent",
        ],
        f"{alphabet[ax_index]}",
        offset_scaling=(0.8, -0.2),
    )

    ax.annotate(
        f"{alphabet[ax_index]}: {float(missing[0]):.1f}\% missing",
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0.1, 0.9),
        textcoords="axes fraction",
        transform=ax.transAxes,
        c="black",
        fontsize=18,
        bbox=dict(
            facecolor="lightgrey", edgecolor="white", boxstyle="round", alpha=0.7
        ),
    )

    ax.set_xlabel("Lag ($\\tau$)")

    print(f"\nStats for interval version {alphabet[ax_index]} as plotted:\n")
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

ax0.set_xscale("log")
ax0.set_xlabel("Lag ($\\tau$)")
ax0.set_ylabel("\% pairs missing", color="grey")
ax0.tick_params(axis="y", colors="grey")
ax0.set_ylim(0, 100)

plt.savefig(
    f"plots/results/{output_path}/test_{spacecraft}_case_study_correcting_{file_index}_{int_index}_{n_bins}_bins.png",
    bbox_inches="tight",
)
