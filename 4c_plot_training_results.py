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

spacecraft = "psp"

plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=16)

# Import all corrected (test) files
n_bins = int(sys.argv[1])
# times_to_gap = params.times_to_gap # removing as will only be using this file locally

data_path_prefix = params.data_path_prefix

# Also do publication-read plots for heatmaps

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
