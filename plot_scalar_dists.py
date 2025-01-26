# MULTI-DISTRIBUTION PLOT
# This script plots the distributions of the slopes, correlation scales, etc., for the different gap handling methods
# Inspired by Fig. 6 (Taylor scales) in Reynolds paper

# For publication, perhaps add asterisks to indicate significance of difference of means


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scikit_posthocs import posthoc_tukey
from scipy.stats import kruskal, levene, shapiro, wilcoxon

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
# Log versions (plot better than densities than using ax.semilogx())
ints["Re_lt_log"] = np.log10(ints["Re_lt"])
ints["Re_lt_log_orig"] = np.log10(ints["Re_lt_orig"])

ints["ttu_log"] = np.log10(ints["ttu"])
ints["ttu_log_orig"] = np.log10(ints["ttu_orig"])
ints["tce_log"] = np.log10(ints["tce"])
ints["tce_log_orig"] = np.log10(ints["tce_orig"])


# Get # unique combinations of file_index and int_index
n_combinations = len(ints.groupby(["file_index", "int_index"]))

print("Plotting and testing distributions of SF-derived stats\n")

print(
    f"{n_combinations} Wind intervals, gapped {len(ints.version.unique())} times each"
)


# Define functions to convert between spectral index and slope of the regression line
def sf_to_psd(x):
    return -(x + 1)


def psd_to_sf(x):
    return -(x - 1)


ints["es_slope"] = ints["slope"].apply(sf_to_psd)
ints["es_slope_orig"] = ints["slope_orig"].apply(sf_to_psd)

variables = ["es_slope", "tce", "ttu_log", "Re_lt_log"]

# Create list of x-labels with usual Latex symbols
xlabels = [
    r"$\beta$",
    r"$\lambda_C$ (lags)",
    r"$\lambda_T$ (lags)",
    r"$Re_{\lambda_T}$",
]
# Create a 2x2 multipanel plot
fig, axes = plt.subplots(2, 2, figsize=(4, 3))

for i, variable in enumerate(variables):
    # Prepare data for each variable
    data_naive = ints[ints.gap_handling == "naive"][variable]
    data_lint = ints[ints.gap_handling == "lint"][variable]
    data_corrected = ints[ints.gap_handling == "corrected_3d"][variable]
    data_true = ints[ints.gap_handling == "naive"][f"{variable}_orig"]

    ax = axes[i // 2, i % 2]

    # Create the KDE plot for each distribution
    sns.kdeplot(
        data_true,
        label=r"\textbf{True}",
        color="lightgrey",
        lw=2,
        ax=ax,
    )
    sns.kdeplot(
        data_corrected,
        label="Corrected",
        color="#1b9e77",
        linestyle="dotted",
        lw=0.8,
        ax=ax,
    )
    sns.kdeplot(
        data_naive,
        label="Naive",
        color="indianred",
        linestyle="dashed",
        ax=ax,
        lw=0.8,
    )
    sns.kdeplot(
        data_lint,
        label="LINT",
        color="black",
        linestyle="dashdot",
        ax=ax,
        lw=0.8,
    )

    # Add labels and title specific to the variable
    ax.set_xlabel(xlabels[i])
    ax.set_yticks([])
    if i % 2 == 0:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("")

    x_annotat = 0.8

    if variable == "Re_lt_log":
        # For getting means of original data
        x_annotat = 0.05
        variable = "Re_lt"
        data_naive = ints[ints.gap_handling == "naive"][variable]
        data_lint = ints[ints.gap_handling == "lint"][variable]
        data_corrected = ints[ints.gap_handling == "corrected_3d"][variable]
        data_true = ints[ints.gap_handling == "naive"][f"{variable}_orig"]

        # Step 4: Customize the x-axis to show original data scale
        ticks = ax.get_xticks()  # Get current ticks on the log scale
        ax.set_xticklabels(
            [f"$10^{{{int(tick)}}}$" for tick in ticks]
        )  # Replace with exponent labels

    if variable == "ttu_log":
        # For getting means of original data
        variable = "ttu"
        data_naive = ints[ints.gap_handling == "naive"][variable]
        data_lint = ints[ints.gap_handling == "lint"][variable]
        data_corrected = ints[ints.gap_handling == "corrected_3d"][variable]
        data_true = ints[ints.gap_handling == "naive"][f"{variable}_orig"]

        # Step 4: Customize the x-axis to show original data scale
        ticks = ax.get_xticks()
        ax.set_xticklabels(
            [f"$10^{{{int(tick)}}}$" for tick in ticks]
        )  # Replace with exponent labels

    # Add annotations of the mean of each distribution
    ax.annotate(
        r"\textbf{" + f"{data_true.mean():.3g}" + "}",
        xy=(x_annotat, 0.85),
        xycoords="axes fraction",
        fontsize=8,
        color="darkgrey",
        ha="left",
    )
    ax.annotate(
        f"{data_corrected.mean():.3g}",
        xy=(x_annotat, 0.7),
        xycoords="axes fraction",
        fontsize=8,
        color="#1b9e77",
        ha="left",
    )
    ax.annotate(
        f"{data_naive.mean():.3g}",
        xy=(x_annotat, 0.55),
        xycoords="axes fraction",
        fontsize=8,
        color="indianred",
        ha="left",
    )
    ax.annotate(
        f"{data_lint.mean():.3g}",
        xy=(x_annotat, 0.4),
        xycoords="axes fraction",
        fontsize=8,
        color="black",
        ha="left",
    )

    if variable == "es_slope":
        # Add vertical line and annotation for K41 prediction
        ax.axvline(-5 / 3, color="mediumblue", linestyle="solid", alpha=0.3, lw=0.5)
        ax.text(
            -5 / 3 - 0.25,
            3,
            "K41",
            va="center",
            ha="left",
            fontsize=7,
            color="mediumblue",
            alpha=0.5,
        )


ax.legend(loc="upper center", bbox_to_anchor=(-0.05, 1.5), ncol=4, fontsize=8)

# Save and display the plot
plt.tight_layout()
plt.subplots_adjust(hspace=1, wspace=0.1)
plt.savefig(f"plots/results/{output_path}/densities.png", bbox_inches="tight", dpi=300)

# 108 std intervals across 40 days of Wind data, gapped 25 times each
# (2700 ints)

# STATISTICAL TEST OF DIFFERENCE OF MEANS
# Perform an ANOVA test to see if the means are significantly different?

for variable in ["slope", "tce", "ttu", "Re_lt"]:

    print(f"\n\nStatistical tests for {variable}\n")

    data_naive = ints[ints.gap_handling == "naive"][variable]
    data_lint = ints[ints.gap_handling == "lint"][variable]
    data_corrected = ints[ints.gap_handling == "corrected_3d"][variable]
    data_true = ints[ints.gap_handling == "naive"][f"{variable}_orig"]

    # Print summary stats of each
    print(f"Summary statistics for {variable}:")
    print("Naive:")
    print(data_naive.describe())
    print("LINT:")
    print(data_lint.describe())
    print("Corrected:")
    print(data_corrected.describe())
    print("True:")
    print(data_true.describe())

    if variable == "slope":

        # Perform a Wilcoxon signed-rank test to see if the means are significantly different from 2/3
        print(
            f"Results of Wilcoxon signed-rank test for difference of mean of True distribution from 2/3 for {variable}:"
        )
        print(wilcoxon(data_true - 2 / 3))
        # WilcoxonResult(statistic=765975.0, pvalue=0.0)

    # First, testing assumptions of ANOVA

    # 1. Homogeneity of variances
    print(f"\nResults of Levene's test for homogeneity of variances for {variable}:")
    print(levene(data_naive, data_lint, data_corrected, data_true))
    # If the p-value is less than 0.05, then the variances are significantly different
    # print("HOMOGENEITY OF VARIANCES NOT SATISFIED")

    # 2. Normality
    print(f"\nResults of Shapiro-Wilk test for normality for {variable}:")
    print(shapiro(data_naive))
    print(shapiro(data_lint))
    print(shapiro(data_corrected))
    print(shapiro(data_true))
    # print("NORMALITY NOT SATISFIED")
    # If the p-value is less than 0.05, then the data is not normally distributed
    # print("\nCANNOT PROCEED WITH ANOVA")

    # Combine all the data into one array
    all_data = pd.DataFrame(
        {
            variable: pd.concat([data_naive, data_lint, data_corrected, data_true]),
            "group": ["naive"] * len(data_naive)
            + ["lint"] * len(data_lint)
            + ["corrected"] * len(data_corrected)
            + ["orig"] * len(data_true),
        }
    )

    # Since we can't perform ANOVA, instead use non-parametric test for difference of means
    # H0: The means of the groups are equal
    # H1: At least one of the means is different
    print(
        f"\nResults of non-parametric Kruskal-Wallis test for difference of means for {variable}:"
    )
    print(kruskal(data_naive, data_lint, data_corrected, data_true))
    # print("REJECT NULL HYPOTHESIS OF EQUAL MEANS")

    # If the Kruskal-Wallis test is significant, then perform a post-hoc test to see which groups are different
    print(f"\nResults of post-hoc Dunn's test for difference of means for {variable}:")
    print(
        posthoc_tukey(
            all_data,
            val_col=variable,
            group_col="group",
        )
    )

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
#     rmsval = np.sqrt(np.mean(ar**2))
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
