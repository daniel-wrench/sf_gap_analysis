# MULTI-DISTRIBUTION PLOT
# This script plots the distributions of the slopes, correlation scales, etc., for the different gap handling methods
# Inspired by Fig. 6 (Taylor scales) in Reynolds paper

# For publication, perhaps add asterisks to indicate significance of difference of means

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.params as params

# from scikit_posthocs import posthoc_dunn
# from scipy.stats import kruskal, levene, shapiro, wilcoxon


plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# Load data
ints = pd.read_csv("ints_gapped_metadata.csv")
# variable = sys.argv[1]  # Change this to the variable you want to analyze
output_path = params.output_path

# Remove any tce values that are less than 0 (due to not getting proper fit)
ints = ints[ints.tce >= 0]

# Get # unique combinations of file_index and int_index
n_combinations = len(ints.groupby(["file_index", "int_index"]))

# data_naive = ints[ints.gap_handling == "naive"][variable]
# data_lint = ints[ints.gap_handling == "lint"][variable]
# data_corrected = ints[ints.gap_handling == "corrected_3d"][variable]
# data_true = ints[ints.gap_handling == "naive"][f"{variable}_orig"]

# # Perform a Wilcoxon signed-rank test to see if the means are significantly different from 2/3
# print(
#     f"Results of Wilcoxon signed-rank test for difference of mean of True distribution from 2/3 for {variable}:"
# )
# print(wilcoxon(data_true - 2 / 3))
# # WilcoxonResult(statistic=765975.0, pvalue=0.0)


# Define functions to convert between spectral index and slope of the regression line
def sf_to_psd(x):
    return -(x + 1)


def psd_to_sf(x):
    return -(x - 1)


variables = ["slope", "tce", "ttu"]
titles = ["Inertial range slope", "Correlation scale", "Taylor scale"]
# Create list of x-labels with usual Latex symbols
xlabels = [r"$\beta$", r"$\lambda_C$ (lags)", r"$\lambda_T$ (lags)"]

# Create a 1x3 multipanel plot
fig, axes = plt.subplots(1, 3, figsize=(7, 3))

for i, variable in enumerate(variables):
    # Prepare data for each variable
    data_naive = ints[ints.gap_handling == "naive"][variable]
    data_lint = ints[ints.gap_handling == "lint"][variable]
    data_corrected = ints[ints.gap_handling == "corrected_3d"][variable]
    data_true = ints[ints.gap_handling == "naive"][f"{variable}_orig"]

    ax = axes[i]

    # Create the KDE plot for each distribution
    sns.kdeplot(
        data_true,
        label="True",  # ({data_true.mean():.3f})
        color="grey",
        lw=2,
        ax=ax,
    )
    sns.kdeplot(
        data_corrected,
        label="Corrected",
        color="#1b9e77",
        linestyle="dotted",
        ax=ax,
    )
    sns.kdeplot(
        data_naive,
        label="Naive",
        color="indianred",
        linestyle="dashed",
        ax=ax,
    )
    sns.kdeplot(
        data_lint,
        label="LINT",
        color="black",
        linestyle="dashdot",
        ax=ax,
    )

    # Add vertical lines at the means of the distributions
    # ax.axvline(data_naive.mean(), color="indianred", linestyle="dashed", lw=0.7)
    # ax.axvline(data_lint.mean(), color="black", linestyle="dashdot", lw=0.7)
    # ax.axvline(data_corrected.mean(), color="#1b9e77", linestyle="dotted", lw=0.7)
    # ax.axvline(data_true.mean(), color="grey", linestyle="solid", lw=0.7)

    # Add labels and title specific to the variable
    ax.set_title(titles[i])
    ax.set_xlabel(xlabels[i])
    ax.set_yticks([])
    if i > 0:
        ax.set_ylabel("")

    if variable == "slope":
        # Create a secondary x-axis for "slope"
        secax = ax.secondary_xaxis("top", functions=(sf_to_psd, psd_to_sf))
        secax.set_xlabel("Equivalent spectral index", fontsize=8)

        # Add vertical line and annotation for K41 prediction
        ax.axvline(2 / 3, color="mediumblue", linestyle="solid", alpha=0.3, lw=0.5)
        ax.text(
            2 / 3 + 0.01,
            3,
            "K41 prediction",
            va="center",
            ha="left",
            fontsize=7,
            color="mediumblue",
            alpha=0.5,
        )

        # ax.set_ylim(1e-2, 1e1)
        # ax.set_xlim(-0.15, 1.25)

    if variable == "ttu":
        ax.set_xlim(0, 30)

axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4)

# Save and display the plot
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.suptitle(
    r"\textbf{Effect of gap handling methods on SF-derived stats} (108 Wind ints gapped 25x each)",
    size=12,
    y=1.05,
)
plt.savefig(f"plots/results/{output_path}/dist_all_variables.pdf", bbox_inches="tight")

# 108 std intervals across 40 days of Wind data, gapped 25 times each
# (2700 ints)


# # STATISTICAL TEST OF DIFFERENCE OF MEANS
# # Perform an ANOVA test to see if the means are significantly different?

# # First, testing assumptions of ANOVA

# # 1. Homogeneity of variances
# print(f"\nResults of Levene's test for homogeneity of variances for {variable}:")
# print(levene(data_naive, data_lint, data_corrected, data_true))
# # If the p-value is less than 0.05, then the variances are significantly different
# # print("HOMOGENEITY OF VARIANCES NOT SATISFIED")

# # 2. Normality
# print(f"\nResults of Shapiro-Wilk test for normality for {variable}:")
# print(shapiro(data_naive))
# print(shapiro(data_lint))
# print(shapiro(data_corrected))
# print(shapiro(data_true))
# # print("NORMALITY NOT SATISFIED")
# # If the p-value is less than 0.05, then the data is not normally distributed
# # print("\nCANNOT PROCEED WITH ANOVA")

# # Combine all the data into one array
# all_data = pd.DataFrame(
#     {
#         variable: pd.concat([data_naive, data_lint, data_corrected, data_true]),
#         "group": ["naive"] * len(data_naive)
#         + ["lint"] * len(data_lint)
#         + ["corrected"] * len(data_corrected)
#         + ["orig"] * len(data_true),
#     }
# )

# # Since we can't perform ANOVA, instead use non-parametric test for difference of means
# # H0: The means of the groups are equal
# # H1: At least one of the means is different
# print(
#     f"\nResults of non-parametric Kruskal-Wallis test for difference of means for {variable}:"
# )
# print(kruskal(data_naive, data_lint, data_corrected, data_true))
# # print("REJECT NULL HYPOTHESIS OF EQUAL MEANS")

# # If the Kruskal-Wallis test is significant, then perform a post-hoc test to see which groups are different
# print(f"\nResults of post-hoc Dunn's test for difference of means for {variable}:")
# print(
#     posthoc_dunn(
#         all_data,
#         val_col=variable,
#         group_col="group",
#     )
# )

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
