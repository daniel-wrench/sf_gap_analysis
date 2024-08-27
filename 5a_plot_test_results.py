import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import src.params as params

np.random.seed(123)  # For reproducibility

dir = "raapoi_test"
times_to_gap = 25

plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=16)

# Import all corrected (test) files
spacecraft = sys.argv[1]
n_bins = sys.argv[2]
# times_to_gap = params.times_to_gap # removing as will only be using this file locally

data_path_prefix = params.data_path_prefix

output_file_path = f"data/processed/{dir}/test_corrected_{spacecraft}_{n_bins}_bins.pkl"

# Read in the file that has just been exported above
with open(output_file_path, "rb") as f:
    data = pickle.load(f)

files_metadata = data["files_metadata"]
ints_metadata = data["ints_metadata"]
# ints = data["ints"]
ints_gapped_metadata = data["ints_gapped_metadata"]
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
colors = ["indianred", "grey", "C0", "purple"]

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
    f"plots/temp/{dir}/test_{spacecraft}_boxplots_{n_bins}_bins.png",
    bbox_inches="tight",
)

# Regression lines
# Paper figures, so removing sub-par 2D correction

custom_order = ["naive", "lint", "corrected_3d"]
colors = ["indianred", "grey", "purple"]

# Make scatterplot of mape vs. missing_percent, coloured by gap handling
palette = dict(zip(custom_order, colors))

# unique_gap_handling = ints_gapped_metadata["gap_handling"].unique()
# unique_gap_handling = ["naive", "lint", "corrected_3d"]
sns.set_style("ticks")
# Plotting the MAPE vs. missing percentage
fig, ax = plt.subplots(
    2,
    len(custom_order) + 1,
    figsize=(10, 5),
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
        alpha=0.3,
        s=10,
        color=palette[gap_handling_method],
        label=gap_handling_method,
        ax=ax[0, i],
        legend=False,
    )

    # Create a secondary axis for the boxplot on the right-hand spine
    ax_right = ax[0, i].inset_axes([1, 0, 0.1, 1], sharey=ax[0, i])
    sns.boxplot(
        data=subset,
        y="mape",
        ax=ax_right,
        orient="v",
        whis=(0, 100),
        color=palette[gap_handling_method],
        linecolor="black",
        linewidth=1.5,
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
        ax=ax[0, -1],
        ci=99,
    )

    sns.scatterplot(
        data=subset,
        x="missing_percent_overall",
        y="slope_ape",
        alpha=0.3,
        s=10,
        color=palette[gap_handling_method],
        label=gap_handling_method,
        ax=ax[1, i],
        legend=False,
    )

    sns.regplot(
        data=subset,
        x="missing_percent_overall",
        y="slope_ape",
        scatter=False,
        color=palette[gap_handling_method],
        label=gap_handling_method,
        order=2,
        ax=ax[1, -1],
        ci=99,
    )


ax[0, 0].set(xlabel="", ylabel="MAPE (\%)", title="Naive")
ax[0, 1].set(xlabel="", ylabel="", title="LINT")
ax[0, 2].set(xlabel="", ylabel="", title="Corrected")
ax[0, 3].set(xlabel="", ylabel="", title="All")

ax[1, 0].set(xlabel="", ylabel="Slope APE (\%)", title="")
ax[1, 1].set(xlabel="", ylabel="", title="")
ax[1, 2].set(xlabel="", ylabel="", title="")
ax[1, 3].set(xlabel="", ylabel="", title="")
# Remove gridlines and plot outlines

# Make one x-axis label for all plots
fig.text(0.5, 0.02, "\% missing", ha="center", va="center")

for i in range(2):
    for j in range(4):
        ax[i, j].grid(False)
        ax[i, j].set_xticks([0, 25, 50, 75, 100])
        ax[i, j].set_xlim(-15, 105)
        ax[i, j].spines["top"].set_visible(False)
        # ax[i, j].spines["left"].set_visible(False)
        ax[i, j].spines["right"].set_visible(False)

# Set the same x-axis limits for all plots
for i in range(4):
    ax[1, i].set_ylim(0, 60)
    ax[1, i].set_xlim(-15, 105)
    ax[0, i].set_ylim(0, 150)

# Remove ticks from all but first column
# for i in range(1, 4):
#     ax[0, i].set_yticks([])
#     ax[1, i].set_yticks([])


ax[0, 0].spines["left"].set_visible(True)
ax[1, 0].spines["left"].set_visible(True)

# Add title
plt.suptitle(
    f"Error vs. \% missing data for the {str.upper(spacecraft)} test set ({n_bins} bins)"
)


plt.savefig(
    f"plots/temp/{dir}/test_{spacecraft}_scatterplots_{n_bins}_bins.png",
    bbox_inches="tight",
)

# Error trendlines

# for gap_handling in sfs_gapped_corrected.gap_handling.unique():
#     sf.plot_error_trend_line(
#         sfs_gapped_corrected[sfs_gapped_corrected["gap_handling"] == gap_handling],
#         estimator="sf_2",
#         title=f"SF estimation error ({gap_handling}) vs. lag and global sparsity",
#         y_axis_log=True,
#     )
#     plt.savefig(
#         f"plots/temp/test_{spacecraft}_error_trend_{gap_handling}_{n_bins}_bins.png",
#         bbox_inches="tight",
#     )
