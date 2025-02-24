import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import src.params as params

plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# Suppress the specific RankWarning from numpy - occurs with fitting slope sometimes
warnings.filterwarnings("ignore", category=np.RankWarning)

np.random.seed(123)  # For reproducibility

run_mode = params.run_mode

# Import all corrected (test) files
spacecraft = "wind"
n_bins = 25

data_path_prefix = params.data_path_prefix

output_file_path = f"results/{run_mode}/test_{spacecraft}_corrected_{n_bins}_bins.pkl"

# Read in the file that has just been exported above
with open(output_file_path, "rb") as f:
    data = pickle.load(f)

files_metadata = data["files_metadata"]
ints_metadata = data["ints_metadata"]
ints_gapped_metadata = data["ints_gapped_metadata"]
# Export as csv
ints_gapped_metadata.to_csv(
    f"results/{run_mode}/test_{spacecraft}_corrected_{n_bins}_bins_ints_gapped_metadata.csv",
    index=False,
)
files_metadata.to_csv(
    f"results/{run_mode}/test_{spacecraft}_files_metadata.csv",
    index=False,
)

ints_gapped_metadata.head()
# Give a unique identifier for each combination of file_index and int_index
ints_gapped_metadata["id"] = (
    ints_gapped_metadata["file_index"].astype(str)
    + "_"
    + ints_gapped_metadata["int_index"].astype(str)
)

# Filtering out bad tces
# SHOUDLN'T BE NEEDED NOW WITH PROPER RETURNING OF NP.NAN RATHER THAN -1
# WHEN CORRELATION SCALE CAN'T BE FOUND (STEP 5)
# ints_gapped_metadata = ints_gapped_metadata[ints_gapped_metadata.tce_orig >= 0]
# ints_gapped_metadata.loc[ints_gapped_metadata.tce == -1, "tce"] = (
#     params.max_lag_prop * params.int_length
# )

# ints_gapped = data["ints_gapped"]
# sfs = data["sfs"]
# sfs_gapped_corrected = data["sfs_gapped_corrected"]

print(
    f"Now plotting test set results for the {len(ints_metadata)} (original) intervals in the {spacecraft} test set"
)


# Assuming ints_gapped_metadata is your DataFrame
# Define the list of columns to plot
columns = ["mpe", "mape", "slope_pe", "slope_ape"]

# Create subplots
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# Flatten the 2D array of axes for easy iteration
axes = axes.flatten()

custom_order = [
    "naive",
    "lint",
    "corrected_3d",
]
colors = ["indianred", "dimgrey", "#1b9e77"]

# Create boxplots for each column
for col, ax in zip(columns, axes):
    data_to_plot = [
        ints_gapped_metadata[ints_gapped_metadata["gap_handling"] == method][col]
        for method in custom_order
    ]
    box = ax.boxplot(data_to_plot, whis=(0, 100), patch_artist=True)

    # Set colors for the boxes

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    # Set colors for the median lines
    median_color = "black"
    for median in box["medians"]:
        median.set_color(median_color)
        median.set_linewidth(2)  # optional: set line width to make it more prominent

    ax.set_title(f"{col}")
    ax.set_ylabel(f"{col}")
    ax.set_xticklabels(custom_order)
    # Add a horizontal at 0 for PE and MPE
    # Set y-limits
    if col == "slope_pe":
        ax.set_ylim(-100, 100)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
    elif col == "slope_ape":
        ax.set_ylim(0, 100)
    elif col == "mpe":
        ax.set_ylim(-100, 100)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
    elif col == "mape":
        ax.set_ylim(0, 100)

# Adjust layout
plt.tight_layout()

plt.suptitle("")  # Remove the default title to avoid overlap
plt.savefig(
    f"results/{run_mode}/plots/test_{spacecraft}_boxplots_{n_bins}_bins.pdf",
    bbox_inches="tight",
)


# -------------------------------------------------------------------------------------#

# Scatterplots of error vs. TGP for each gap handling method

# (Not including sub-par 2D corrected results)

custom_order = [
    "naive",
    "lint",
    "corrected_3d",
]
colors = ["indianred", "dimgrey", "#1b9e77"]
ylims = {
    "mape": (0, 70),
    "slope_ape": (0, 50),
    "tce_ape": (0, 200),
    "ttu_ape": (0, 100),
}
# Make scatterplot of mape vs. missing_percent, coloured by gap handling
palette = dict(zip(custom_order, colors))

mean_y_coord = [0.5, 0.6, 0.7, 0.8, 0.9]

# unique_gap_handling = ints_gapped_metadata["gap_handling"].unique()
# unique_gap_handling = ["naive", "lint", "corrected_3d"]

for error_metric in ["mape", "slope_ape", "tce_ape", "ttu_ape"]:
    fig, ax = plt.subplots(
        1,
        4,
        figsize=(8, 2),
        sharex="col",
        sharey="row",
        tight_layout=True,
    )
    plt.subplots_adjust(wspace=0)

    ax = ax.flatten()

    # Add regression lines for each group

    for i, gap_handling_method in enumerate(custom_order):
        subset = ints_gapped_metadata[
            ints_gapped_metadata["gap_handling"] == gap_handling_method
        ]
        sns.scatterplot(
            data=subset,
            x="missing_percent_overall",
            y=error_metric,
            alpha=0.1,
            s=10,
            color=palette[gap_handling_method],
            label=gap_handling_method,
            ax=ax[i],
            legend=False,
        )

        # Create a secondary axis for the boxplot on the right-hand spine
        ax_right = ax[i].inset_axes([1, 0, 0.1, 1], sharey=ax[i])
        sns.boxplot(
            data=subset,
            y=error_metric,
            ax=ax_right,
            orient="v",
            whis=(0, 100),
            color=palette[gap_handling_method],
            linecolor="black",
            linewidth=1.2,
        )

        # Hide the y-axis labels of the boxplot to avoid duplication
        ax_right.yaxis.set_visible(False)

        # # Optional: Remove the x-axis labels and ticks of the boxplot
        # ax_right.set_xticks([])
        # ax_right.set_xlabel("")

        ax_right.spines["left"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        ax_right.spines["top"].set_visible(False)

        sns.regplot(
            data=subset,
            x="missing_percent_overall",
            y=error_metric,
            scatter=False,
            color=palette[gap_handling_method],
            label=gap_handling_method,
            order=1,
            ax=ax[-1],
            ci=None,
            line_kws={"linewidth": 0.8},  # Set the line width to be thinner
        )

        # Add annotation for the mean value
        ax[-1].annotate(
            f"MAPE: {subset[error_metric].mean():.1f}",
            xy=(0.8, mean_y_coord[i]),
            xycoords="axes fraction",
            fontsize=8,
            c=palette[gap_handling_method],
        )

    # Move titles to inside top each plot
    for i, title in enumerate(
        [
            "Naive",
            "LINT",
            "Corrected 3D",
            "Regression lines",
        ]
    ):
        ax[i].text(
            0.5,
            0.98,
            title,
            transform=ax[i].transAxes,
            verticalalignment="top",
            horizontalalignment="center",
        )

    # ax[1, 0].set(xlabel="", ylabel="Slope APE (\%)", title="")
    # ax[1, 1].set(xlabel="", ylabel="", title="")
    # ax[1, 2].set(xlabel="", ylabel="", title="")
    # ax[1, 3].set(xlabel="", ylabel="", title="")
    # Remove gridlines and plot outlines

    # Make one x-axis label for all plots
    fig.text(0.5, 0.00, "TGP (\%)", ha="center", va="center")

    # for i in range(2):
    for axis in ax:
        axis.grid(False)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_xlim(-15, 105)
        axis.set(xlabel="", ylabel="")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        axis.set_ylim(ylims[error_metric])

    plt.suptitle(f"{error_metric} vs. TGP for {spacecraft} test set")
    plt.savefig(
        f"results/{run_mode}/plots/test_{spacecraft}_scatterplots_{n_bins}_bins_{error_metric}.png",
        bbox_inches="tight",
        dpi=300,
    )

# Error trendlines (REQUIRE FULL CORRECTED SFS, NOT CURRENTLY OUTPUT FROM HPC)

# for gap_handling in sfs_gapped_corrected.gap_handling.unique():
#     sf.plot_error_trend_line(
#         sfs_gapped_corrected[sfs_gapped_corrected["gap_handling"] == gap_handling],
#         estimator="sf_2",
#         title=f"SF estimation error ({gap_handling.upper()}) vs. lag and global sparsity",
#         y_axis_log=True,
#     )
#     plt.savefig(
#         f"plots/results/test_{spacecraft}_error_trend_{gap_handling.upper()}_{n_bins}_bins.png",
#         bbox_inches="tight",
#     )
