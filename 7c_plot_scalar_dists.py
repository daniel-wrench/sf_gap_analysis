# MULTI-DISTRIBUTION PLOT
# This script plots the distributions of the slopes, correlation scales, etc., for the different gap handling methods
# Inspired by Fig. 6 (Taylor scales) in Reynolds paper

# For publication, perhaps add asterisks to indicate significance of difference of medians
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.params as params

plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# Load data
ints = pd.read_csv("ints_gapped_metadata.csv")
output_path = params.output_path


# Remove any tce values that are less than 0 (due to not getting proper fit)
ints = ints[ints.tce_orig >= 0]
ints = ints[ints.tce >= 0]

# Checking proportions of above filters
len(ints)
# 44100
len(ints[ints.tce < 0])
# 2699
len(ints[ints.tce_orig < 0])
# 700

# 1.6% of original SFs and 6.1% of the gapped SFs have negative tce values

# Calculate ttu-based Reynolds number
ints["Re_lt"] = 27 * (ints["tce"] / ints["ttu"]) ** 2
ints["Re_lt_orig"] = 27 * (ints["tce_orig"] / ints["ttu_orig"]) ** 2

# Calculate logarithms of log-distributed stats
# (this plots better densities than using ax.semilogx())
ints["Re_lt_log"] = np.log10(ints["Re_lt"])
ints["Re_lt_log_orig"] = np.log10(ints["Re_lt_orig"])

ints["ttu_log"] = np.log10(ints["ttu"])
ints["ttu_log_orig"] = np.log10(ints["ttu_orig"])

ints["tce_log"] = np.log10(ints["tce"])
ints["tce_log_orig"] = np.log10(ints["tce_orig"])


# Convert between spectral index and slope of the regression line
def sf_to_psd(x):
    return -(x + 1)


def psd_to_sf(x):
    return -(x - 1)


ints["es_slope"] = ints["slope"].apply(sf_to_psd)
ints["es_slope_orig"] = ints["slope_orig"].apply(sf_to_psd)


print("Plotting and testing distributions of SF-derived stats\n")

# Get # unique combinations of file_index and int_index
n_combinations = len(ints.groupby(["file_index", "int_index"]))

print(
    f"{n_combinations} Wind intervals, gapped {len(ints.version.unique())} times each"
)

# Define the variables to plot
variables = ["es_slope", "tce", "ttu", "Re_lt"]

# Create list of x-labels with usual Latex symbols
xlabels = [
    r"$\beta$",
    r"$\lambda_C$ (lags)",
    r"$\lambda_T$ (lags)",
    r"$Re_{\lambda_T}$",
]

# Create column "tgp_bins" to store the bins of the TGP
bin_labels = ["0-25", "25-50", "50-75", "75-100"]
ints["tgp_bin"] = pd.cut(
    ints["missing_percent_overall"],
    bins=[0, 25, 50, 75, 100],
    labels=bin_labels,
)

# Loop over each bin of missing data, including the full dataset
for bin in bin_labels + ["all_data"]:

    if bin == "all_data":
        # Condition data for the full dataset
        data_naive = ints[ints.gap_handling == "naive"]
        data_lint = ints[ints.gap_handling == "lint"]
        data_corrected = ints[ints.gap_handling == "corrected_3d"]
        data_true = ints[ints.gap_handling == "naive"]
    else:
        # Condition data based on gap handling method and TGP bin
        data_naive = ints[(ints.gap_handling == "naive") & (ints.tgp_bin == bin)]
        data_lint = ints[(ints.gap_handling == "lint") & (ints.tgp_bin == bin)]
        data_corrected = ints[
            (ints.gap_handling == "corrected_3d") & (ints.tgp_bin == bin)
        ]
        data_true = ints[(ints.gap_handling == "naive") & (ints.tgp_bin == bin)]

    # Create a 2x2 multipanel plot
    fig, axes = plt.subplots(1, 4, figsize=(6, 1.5))

    for i, variable in enumerate(variables):
        variable_to_plot = variable
        ax = axes[i]

        print(f"Plotting {variable} for bin {bin}")

        if variable == "Re_lt":
            variable_to_plot = "Re_lt_log"
        if variable == "ttu":
            variable_to_plot = "ttu_log"
        if variable == "tce":
            variable_to_plot = "tce_log"

        # Create the KDE plot for each distribution
        sns.kdeplot(
            data_true[f"{variable_to_plot}_orig"],
            label=r"\textbf{True}",
            color="lightgrey",
            lw=2,
            ax=ax,
        )
        sns.kdeplot(
            data_corrected[variable_to_plot],
            label="Corrected",
            color="#1b9e77",
            linestyle="dotted",
            lw=0.8,
            ax=ax,
        )
        sns.kdeplot(
            data_naive[variable_to_plot],
            label="Naive",
            color="indianred",
            linestyle="dashed",
            ax=ax,
            lw=0.8,
        )
        sns.kdeplot(
            data_lint[variable_to_plot],
            label="LINT",
            color="black",
            linestyle="dashdot",
            ax=ax,
            lw=0.8,
        )

        # Add labels and title specific to the variable
        ax.set_xlabel(xlabels[i])
        ax.set_ylim(0, 3.7)
        ax.set_yticks([])
        # if i % 2 == 0:
        #     ax.set_ylabel("Density")
        # else:
        ax.set_ylabel("")

        x_annotat = 0.05

        if variable != "es_slope":

            if variable == "tce":
                xmin, xmax = 1.3, 3.6

            elif variable == "ttu":
                xmin, xmax = 0, 2.3

            elif variable == "Re_lt":
                xmin, xmax = 2.2, 6.7

            ax.set_xlim(xmin, xmax)
            ticks = np.arange(
                math.ceil(xmin), math.floor(xmax) + 1, 1
            )  # Creates ticks from 2 to 8 with step size 1
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"$10^{{{int(x)}}}$" for x in ax.get_xticks()])

        # Add annotations of the median of each distribution
        ax.annotate(
            r"\textbf{" + f"{data_true[variable + '_orig'].median():.3g}" + "}",
            xy=(x_annotat, 0.85),
            xycoords="axes fraction",
            fontsize=7,
            color="darkgrey",
            ha="left",
        )
        ax.annotate(
            f"{data_corrected[variable].median():.3g}",
            xy=(x_annotat, 0.7),
            xycoords="axes fraction",
            fontsize=7,
            color="#1b9e77",
            ha="left",
        )
        ax.annotate(
            f"{data_naive[variable].median():.3g}",
            xy=(x_annotat, 0.55),
            xycoords="axes fraction",
            fontsize=7,
            color="indianred",
            ha="left",
        )
        ax.annotate(
            f"{data_lint[variable].median():.3g}",
            xy=(x_annotat, 0.4),
            xycoords="axes fraction",
            fontsize=7,
            color="black",
            ha="left",
        )

        if variable == "es_slope":
            # Add vertical line and annotation for K41 prediction
            ax.axvline(-5 / 3, color="mediumblue", linestyle="solid", alpha=0.3, lw=0.5)
            ax.text(
                -5 / 3 - 0.4,
                2.8,
                r"\textit{K41}",
                va="center",
                ha="left",
                fontsize=7,
                color="mediumblue",
                alpha=0.5,
            )

            ax.set_xlim(-2.8, -0.8)

    if bin == "all_data":
        plt.suptitle("Full dataset", x=-0.001, y=0.7, ha="right")
    else:
        plt.suptitle(f"{bin}\% missing", x=-0.001, y=0.7, ha="right")

    # Tighten layout and adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(
        hspace=0, wspace=0.1, top=0.77
    )  # Adjust `top` to make space for the legend
    plt.show()
    # plt.savefig(
    #     f"plots/results/{output_path}/densities_{bin}.png",
    #     bbox_inches="tight",
    #     dpi=300,
    # )

    # 108 std intervals across 40 days of Wind data, gapped 25 times each
    # (2700 ints)

# ------------------------------------------------------------------------------

# TULASI'S CODE TO CREATE A DENSITY FROM A HISTOGRAM
# Doesn't produce very smooth results, so I used seaborn's kdeplot instead

# def normhist(ar, min=99999, max=99999, nbins=100, inc=0, ax=0):
#     if len(ar) == 0:
#         print("No array provided! Exiting!")
#         return
#     # If PDF of increment, then increment the array
#     if inc > 0:
#         ar = ar - np.roll(ar, inc, axis=ax)
#     # Find the total length of data set
#     arsize = reduce(operator.mul, np.shape(ar), 1)
#     # Find the RMS value of data set and normalize to it.
#     rmsval = np.sqrt(np.median(ar**2))
#     if rmsval != 0:
#         ar = ar / rmsval
#     # Reshape the array to 1D & sort it.
#     arr = np.reshape(ar, arsize)
#     np.ndarray.sort(arr, kind="heapsort")
#     # Empty arrays for output
#     if min == 99999:
#         min = ar.min()
#     if max == 99999:
#         max = ar.max()
#     bins = np.linspace(min, max, nbins)
#     dbin = bins[1] - bins[0]
#     normhist = np.zeros(nbins)
#     # Rescale everything to have zero at min
#     arr = arr - min
#     # Fill the bins
#     for i in range(len(arr)):
#         j = int(np.floor(arr[i] / dbin))
#         normhist[j] = normhist[j] + 1
#     normhist = normhist / (arsize * dbin)
#     return bins, normhist


# # Create histograms for all the different slope
# bins_naive, hist_naive = normhist(slopes_naive.values, nbins=15)
# bins_lint, hist_lint = normhist(slopes_lint.values, nbins=15)
# bins_corrected, hist_corrected = normhist(slopes_corrected.values, nbins=15)
# bins_true, hist_true = normhist(slopes_true.values, nbins=15)

# # Plot the histograms
# plt.figure(figsize=(5, 2.5))
# plt.plot(bins_true, hist_true, label="True", color="grey")
# plt.plot(bins_corrected, hist_corrected, label="Corrected", color="#1b9e77")
# plt.plot(bins_naive, hist_naive, label="Naive", color="indianred")
# plt.plot(bins_lint, hist_lint, label="LINT", color="black")

# # Add labels and title
# plt.xlabel("slope of fitted regression line to log-log SF")
# plt.ylabel("Density")
# plt.legend()
# plt.semilogy()
# # plt.ylim(1e-2, 1e1)
# plt.show()
