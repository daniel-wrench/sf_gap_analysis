import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import src.params as params
import warnings
import pandas as pd
import src.sf_funcs as sf


# Suppress the specific RankWarning from numpy - occurs with fitting slope sometimes
warnings.filterwarnings("ignore", category=np.RankWarning)

np.random.seed(123)  # For reproducibility

output_path = params.output_path
times_to_gap = params.times_to_gap

plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=15)

# Import all corrected (test) files
spacecraft = sys.argv[1]
n_bins = int(sys.argv[2])
# times_to_gap = params.times_to_gap # removing as will only be using this file locally

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
colors = ["indianred", "grey", "C0", "#1b9e77"]

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
    f"plots/results/{output_path}/test_{spacecraft}_boxplots_{n_bins}_bins.png",
    bbox_inches="tight",
)

# Regression lines
# Paper figures, so removing sub-par 2D correction

custom_order = ["naive", "lint", "corrected_3d"]
colors = ["indianred", "grey", "#1b9e77"]

# Make scatterplot of mape vs. missing_percent, coloured by gap handling
palette = dict(zip(custom_order, colors))

# unique_gap_handling = ints_gapped_metadata["gap_handling"].unique()
# unique_gap_handling = ["naive", "lint", "corrected_3d"]
sns.set_style("ticks")
# Plotting the MAPE vs. missing percentage
fig, ax = plt.subplots(
    1,
    len(custom_order) + 1,
    figsize=(10, 3),
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
        ax=ax[-1],
        ci=99,
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


ax[0].set(xlabel="", ylabel="MAPE (\%)", title="Naive")
ax[1].set(xlabel="", ylabel="", title="LINT")
ax[2].set(xlabel="", ylabel="", title="Corrected")
ax[3].set(xlabel="", ylabel="", title="All")

# ax[1, 0].set(xlabel="", ylabel="Slope APE (\%)", title="")
# ax[1, 1].set(xlabel="", ylabel="", title="")
# ax[1, 2].set(xlabel="", ylabel="", title="")
# ax[1, 3].set(xlabel="", ylabel="", title="")
# Remove gridlines and plot outlines

# Make one x-axis label for all plots
fig.text(0.5, 0.00, "\% missing", ha="center", va="center")

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
plt.suptitle(
    f"Error vs. \% missing data for the {spacecraft.upper()} test set ({n_bins} bins)"
)


plt.savefig(
    f"plots/results/{output_path}/test_{spacecraft}_scatterplots_{n_bins}_bins.png",
    bbox_inches="tight",
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


# Also do publication-read plots for heatmaps (as least the LINT versions; we haven't output the correction
# lookup for the naive versions, but we do have plots of these on the HPC from step 2b)

# Below is copied directly from 4a_finalise_correction.py
for dim in [2, 3]:
    if dim == 2:  # read both naive and LINT versions
        with open(
            f"data/corrections/{output_path}/correction_lookup_{dim}d_{n_bins}_bins_naive.pkl",
            "rb",
        ) as f:
            correction_lookup_naive = pickle.load(f)

        with open(
            f"data/corrections/{output_path}/correction_lookup_{dim}d_{n_bins}_bins_lint.pkl",
            "rb",
        ) as f:
            correction_lookup_lint = pickle.load(f)

        xedges = correction_lookup_lint["xedges"]
        yedges = correction_lookup_lint["yedges"]
        pe_mean_lint = correction_lookup_lint["pe_mean"]
        pe_mean_naive = correction_lookup_naive["pe_mean"]

        fig, ax = plt.subplots(figsize=(8, 3), ncols=2, sharey=True)
        plt.grid(False)

        # Plot the heatmap for the first column
        sc = ax[0].pcolormesh(
            xedges,
            yedges,
            pe_mean_naive.T,
            cmap="bwr",
        )
        ax[0].grid(False)
        ax[0].set_xlabel("Lag ($\\tau$)")
        ax[0].set_ylabel("\% missing")
        ax[0].set_title("Naive")
        ax[0].set_facecolor("black")
        ax[0].set_xscale("log")

        # Plot the heatmap for the second column
        sc2 = ax[1].pcolormesh(
            xedges,
            yedges,
            pe_mean_lint.T,
            cmap="bwr",
        )
        ax[1].grid(False)
        ax[1].set_xlabel("Lag ($\\tau$)")
        ax[1].set_title("LINT")
        ax[1].set_facecolor("black")
        ax[1].set_xscale("log")

        cb = plt.colorbar(sc, cax=ax[1].inset_axes([1.05, 0, 0.03, 1]))
        sc.set_clim(-100, 100)
        sc2.set_clim(-100, 100)
        cb.set_label("\% error")
        plt.subplots_adjust(wspace=0.108)

        plt.savefig(
            f"plots/results/{output_path}/train_heatmap_{n_bins}bins_2d.png",
            bbox_inches="tight",
        )
        plt.close()

    elif dim == 3:  # read only LINT versions
        with open(
            f"data/corrections/{output_path}/correction_lookup_{dim}d_{n_bins}_bins_lint.pkl",
            "rb",
        ) as f:
            correction_lookup = pickle.load(f)

        xedges = correction_lookup["xedges"]
        yedges = correction_lookup["yedges"]
        zedges = correction_lookup["zedges"]
        pe_mean = correction_lookup["pe_mean"]

        # Define the number of columns (you can adjust this as desired)
        n_cols = 5  # Number of columns per row
        n_rows = (n_bins + n_cols - 1) // n_cols  # Calculate number of rows needed

        # Create subplots with multiple rows and columns
        fig, ax = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 2, n_rows * 2),
            sharex=True,
            sharey=True,
        )
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        plt.grid(False)

        # Flatten the axis array to simplify indexing
        ax = ax.flatten()

        for i in range(n_bins):
            ax[i].grid(False)
            c = ax[i].pcolormesh(
                xedges,
                yedges,
                pe_mean[:, :, i],
                cmap="bwr",
            )
            c.set_clim(-100, 100)
            ax[i].set_title(
                f"Power bin {i+1}/{n_bins}".format(np.round(zedges[i], 2)), fontsize=13
            )
            ax[i].set_facecolor("black")
            ax[i].semilogx()

            # Remove y-axis labels for all but the first column plots
            if i % n_cols == 0:
                ax[i].set_ylabel("\% missing")
        fig.text(
            0.5, 0.00, "Lag ($\\tau$)", ha="center", va="center", fontsize=17
        )  # Shared x-axis label
        # Now do the same but for an overall plot title
        plt.suptitle(
            f"Error vs. \% missing data and lag for the {spacecraft.upper()} training set ({n_bins} bins)",
            fontsize=17,
            y=1.02,
        )
        # Hide any extra subplots if n_bins is not a multiple of n_cols
        for j in range(n_bins, len(ax)):
            fig.delaxes(ax[j])

        # Add a color bar on the right-hand side of the figure, stretching down the entire height
        cbar_ax = fig.add_axes(
            [0.92, 0.105, 0.02, 0.78]
        )  # [left, bottom, width, height] to cover full height
        cb = plt.colorbar(c, cax=cbar_ax)  # Attach the color bar to the last heatmap
        cb.set_label("MPE")  # Optional: Label the color bar

        plt.savefig(
            f"plots/results/{output_path}/train_heatmap_{n_bins}bins_3d_lint_power.png",
            bbox_inches="tight",
        )
        plt.close()

        # Create subplots with multiple rows and columns
        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), sharex=True, sharey=True
        )
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.grid(False)

        # Flatten the axis array to simplify indexing
        ax = ax.flatten()

        for i in range(n_bins):
            ax[i].grid(False)
            c = ax[i].pcolormesh(
                yedges,
                zedges,
                pe_mean[i, :, :],
                cmap="bwr",
            )
            c.set_clim(-100, 100)
            plt.title("Distribution of missing proportion and lag")
            ax[i].set_title(f"Lag bin {i+1}/{n_bins}".format(np.round(zedges[i], 2)))
            ax[i].set_facecolor("black")
            ax[i].semilogy()

            # Remove y-axis labels for all but the first column plots
            if i % n_cols == 0:
                ax[i].set_ylabel("Power")
        fig.text(
            0.5, 0.00, "\% missing", ha="center", va="center", fontsize=17
        )  # Shared x-axis label
        # Hide any extra subplots if n_bins is not a multiple of n_cols
        for j in range(n_bins, len(ax)):
            fig.delaxes(ax[j])

        # Add a color bar on the right-hand side of the figure, stretching down the entire height
        cbar_ax = fig.add_axes(
            [0.92, 0.1, 0.02, 0.8]
        )  # [left, bottom, width, height] to cover full height
        cb = plt.colorbar(c, cax=cbar_ax)  # Attach the color bar to the last heatmap
        cb.set_label("MPE")  # Optional: Label the color bar

        plt.savefig(
            f"plots/results/{output_path}/train_heatmap_{n_bins}bins_3d_lint_lag.png",
            bbox_inches="tight",
        )
        plt.close()

        # Create subplots with multiple rows and columns
        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), sharex=True, sharey=True
        )
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.grid(False)

        # Flatten the axis array to simplify indexing
        ax = ax.flatten()

        for i in range(n_bins):
            ax[i].grid(False)
            c = ax[i].pcolormesh(
                xedges,
                zedges,
                pe_mean[:, i, :],
                cmap="bwr",
            )
            c.set_clim(-100, 100)
            ax[i].set_facecolor("black")
            ax[i].semilogx()
            ax[i].semilogy()
            plt.title("Distribution of missing proportion and lag")
            ax[i].set_title(
                f"Missing prop bin {i+1}/{n_bins}".format(np.round(zedges[i], 2))
            )

            # Remove y-axis labels for all but the first column plots
            if i % n_cols == 0:
                ax[i].set_ylabel("Power")
        fig.text(
            0.5, 0.00, "Lag", ha="center", va="center", fontsize=17
        )  # Shared x-axis label
        # Hide any extra subplots if n_bins is not a multiple of n_cols
        for j in range(n_bins, len(ax)):
            fig.delaxes(ax[j])

        # Add a color bar on the right-hand side of the figure, stretching down the entire height
        cbar_ax = fig.add_axes(
            [0.92, 0.1, 0.02, 0.8]
        )  # [left, bottom, width, height] to cover full height
        cb = plt.colorbar(c, cax=cbar_ax)  # Attach the color bar to the last heatmap
        cb.set_label("MPE")  # Optional: Label the color bar

        plt.savefig(
            f"plots/results/{output_path}/train_heatmap_{n_bins}bins_3d_lint_missing.png",
            bbox_inches="tight",
        )
        plt.close()


# Also plot error trendlines for subset of training results
sfs_gapped = pd.read_pickle("data/processed/psp_train_sfs_gapped.pkl")

sf.plot_error_trend_line(
    sfs_gapped,
    estimator="sf_2",
)
plt.savefig(
    f"plots/results/{output_path}/train_psp_error_trend.png",
    bbox_inches="tight",
)
plt.close()
