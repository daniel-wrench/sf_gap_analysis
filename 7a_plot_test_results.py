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

output_path = params.output_path
times_to_gap = params.times_to_gap

# Import all corrected (test) files
spacecraft = "wind"
n_bins = 25

data_path_prefix = params.data_path_prefix

output_file_path = (
    f"data/corrections/{output_path}/test_corrected_{spacecraft}_{n_bins}_bins.pkl"
)

# Read in the file that has just been exported above
with open(output_file_path, "rb") as f:
    data = pickle.load(f)

files_metadata = data["files_metadata"]
ints_metadata = data["ints_metadata"]
# ints = data["ints"]
ints_gapped_metadata = data["ints_gapped_metadata"]
# Export as csv
ints_gapped_metadata.to_csv("ints_gapped_metadata.csv")
files_metadata.to_csv("files_metadata.csv")


# ints_gapped = data["ints_gapped"]
# fs = data["sfs"]
# sfs_gapped_corrected = data["sfs_gapped_corrected"]

print(
    f"Now plotting test set results for the {len(ints_metadata)}x{times_to_gap} intervals in the {spacecraft} test set"
)


# Assuming ints_gapped_metadata is your DataFrame
# Define the list of columns to plot
columns = ["mpe", "mape", "slope_pe", "slope_ape"]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Flatten the 2D array of axes for easy iteration
axes = axes.flatten()

custom_order = ["naive", "lint", "corrected_2d", "corrected_3d"]
colors = ["indianred", "dimgrey", "C0", "#1b9e77"]

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
    f"plots/results/{output_path}/test_{spacecraft}_boxplots_{n_bins}_bins.pdf",
    bbox_inches="tight",
)

# Regression lines
# Paper figures, so removing sub-par 2D correction

custom_order = ["naive", "lint", "corrected_3d"]
colors = ["indianred", "dimgrey", "#1b9e77"]

# Make scatterplot of mape vs. missing_percent, coloured by gap handling
palette = dict(zip(custom_order, colors))

# unique_gap_handling = ints_gapped_metadata["gap_handling"].unique()
# unique_gap_handling = ["naive", "lint", "corrected_3d"]

# Plotting the MAPE vs. missing percentage
fig, ax = plt.subplots(
    1,
    len(custom_order) + 1,
    figsize=(7, 2),
    sharex="col",
    sharey="row",
    tight_layout=True,
)
plt.subplots_adjust(wspace=0)

# Add regression lines for each group

for i, gap_handling_method in enumerate(custom_order):
    subset = ints_gapped_metadata[
        ints_gapped_metadata["gap_handling"] == gap_handling_method
    ]
    sns.scatterplot(
        data=subset,
        x="missing_percent_overall",
        y="mape",
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
        y="mape",
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
        y="mape",
        scatter=False,
        color=palette[gap_handling_method],
        label=gap_handling_method,
        order=2,
        ax=ax[-1],
        ci=99,
        line_kws={"linewidth": 0.8},  # Set the line width to be thinner
    )

    # sns.scatterplot(
    #     data=subset,
    #     x="missing_percent_overall",
    #     y="slope_ape",
    #     alpha=0.3,
    #     s=10,
    #     color=palette[gap_handling_method],
    #     label=gap_handling_method,
    #     ax=ax[1, i],
    #     legend=False,
    # )

    # sns.regplot(
    #     data=subset,
    #     x="missing_percent_overall",
    #     y="slope_ape",
    #     scatter=False,
    #     color=palette[gap_handling_method],
    #     label=gap_handling_method,
    #     order=2,
    #     ax=ax[1, -1],
    #     ci=99,
    # )


# Move titles to inside top each plot
for i, title in enumerate(["Naive", "LINT", "Corrected", "Regression lines"]):
    ax[i].text(
        0.5,
        0.98,
        title,
        transform=ax[i].transAxes,
        verticalalignment="top",
        horizontalalignment="center",
    )

ax[0].set(xlabel="", ylabel="MAPE (\%)")
ax[1].set(xlabel="", ylabel="")
ax[2].set(xlabel="", ylabel="")
ax[3].set(xlabel="", ylabel="")

# ax[1, 0].set(xlabel="", ylabel="Slope APE (\%)", title="")
# ax[1, 1].set(xlabel="", ylabel="", title="")
# ax[1, 2].set(xlabel="", ylabel="", title="")
# ax[1, 3].set(xlabel="", ylabel="", title="")
# Remove gridlines and plot outlines

# Make one x-axis label for all plots
fig.text(0.5, 0.00, "TGP (\%)", ha="center", va="center")

# for i in range(2):
for j in range(4):
    ax[j].grid(False)
    ax[j].set_xticks([0, 25, 50, 75, 100])
    ax[j].set_xlim(-15, 105)
    ax[j].spines["top"].set_visible(False)
    # ax[j].spines["left"].set_visible(False)
    ax[j].spines["right"].set_visible(False)

    ax[j].set_ylim(0, 60)
    ax[j].set_xlim(-15, 105)
    ax[j].set_ylim(0, 150)

# Remove ticks from all but first column
# for i in range(1, 4):
#     ax[i].set_yticks([])
#     ax[1, i].set_yticks([])


ax[0].spines["left"].set_visible(True)
# ax[1, 0].spines["left"].set_visible(True)

# Add title
# plt.suptitle(f"Error vs. \% missing data for the Wind test set ({n_bins} bins)")
# plt.show()
plt.savefig(
    f"plots/results/{output_path}/test_{spacecraft}_scatterplots_{n_bins}_bins.png",
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
