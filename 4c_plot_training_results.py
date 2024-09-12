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
plt.rc("font", family="serif", serif="Computer Modern", size=13)

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

        fig, ax = plt.subplots(figsize=(5, 2), ncols=2, sharey=True)
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

        # MISSING VS LAG, BY POWER BIN
        fig, ax = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 2, n_rows * 2),
            sharex=True,
            sharey=True,
        )
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        plt.grid(False)
        plt.suptitle(
            r"3D error heatmap: trend with increasing $\mathbf{power}$",
            fontsize=17,
            y=0.98,
        )

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
                f"({np.round(zedges[i], 2)},{np.round(zedges[i+1], 2)})",
                fontsize=13,
            )
            ax[i].set_facecolor("black")
            ax[i].semilogx()

        fig.text(
            0.5, 0.05, "Lag ($\\tau$)", ha="center", va="center", fontsize=17
        )  # Shared x-axis label
        fig.text(
            0.07,
            0.5,
            "\% missing",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=17,
        )  # Shared y-axis label

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

        # POWER VS % MISSING, BY LAG BIN
        fig, ax = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 2, n_rows * 2),
            sharex=True,
            sharey=True,
        )
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        plt.grid(False)
        plt.suptitle(
            r"3D error heatmap: trend with increasing $\mathbf{lag}$",
            fontsize=17,
            y=0.98,  # Was 1.02 for 2 rows
        )

        # Flatten the axis array to simplify indexing
        ax = ax.flatten()

        # Format lag bin edges to integers
        formatted_xedges = [f"{x:.0f}".rstrip("0").rstrip(".") for x in xedges]

        for i in range(n_bins):
            ax[i].grid(False)
            c = ax[i].pcolormesh(
                yedges,
                zedges,
                pe_mean[i, :, :],
                cmap="bwr",
            )
            c.set_clim(-100, 100)
            ax[i].set_title(
                f"({formatted_xedges[i]},{formatted_xedges[i+1]})",
                fontsize=13,
            )
            ax[i].set_facecolor("black")
            ax[i].semilogy()

        fig.text(
            0.5, 0.05, "\% missing", ha="center", va="center", fontsize=17
        )  # Shared x-axis label, was 0.00 y-val for 2 rows
        fig.text(
            0.07,
            0.5,
            "Power",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=17,
        )  # Shared y-axis label

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
            f"plots/results/{output_path}/train_heatmap_{n_bins}bins_3d_lint_lag.png",
            bbox_inches="tight",
        )
        plt.close()

        # POWER VS LAG, BIN % MISSING BIN
        fig, ax = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 2, n_rows * 2),
            sharex=True,
            sharey=True,
        )
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        plt.grid(False)
        plt.suptitle(
            r"3D error heatmap: trend with increasing \% $\mathbf{missing}$",
            fontsize=17,
            y=0.98,
        )
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
            ax[i].set_title(
                f"({np.round(yedges[i], 2)},{np.round(yedges[i+1], 2)})",
                fontsize=13,
            )
            ax[i].set_facecolor("black")
            ax[i].semilogx()
            ax[i].semilogy()

        fig.text(
            0.5, 0.05, "Lag ($\\tau$)", ha="center", va="center", fontsize=17
        )  # Shared x-axis label
        fig.text(
            0.07,
            0.5,
            "Power",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=17,
        )  # Shared y-axis label

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
