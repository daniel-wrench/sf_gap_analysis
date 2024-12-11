# SLOPE HISTOGRAM PLOT
# This script creates a histogram of the slopes of the regression lines fitted to the log-log structure functions.


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scikit_posthocs import posthoc_dunn
from scipy.stats import kruskal, levene, shapiro, wilcoxon

plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# Load data
ints = pd.read_csv("gapped_ints_metadata.csv")

slopes_naive = ints[ints.gap_handling == "naive"].slope
slopes_lint = ints[ints.gap_handling == "lint"].slope
slopes_corrected = ints[ints.gap_handling == "corrected_3d"].slope
slopes_true = ints[ints.gap_handling == "naive"].slope_orig

# Perform a Wilcoxon signed-rank test to see if the means are significantly different from 2/3
print(
    "Results of Wilcoxon signed-rank test for difference of mean of True distribution from 2/3:"
)
print(wilcoxon(slopes_true - 2 / 3))
# WilcoxonResult(statistic=765975.0, pvalue=0.0)


# Define functions to convert between spectral index and slope of the regression line
def sf_to_psd(x):
    return -(x + 1)


def psd_to_sf(x):
    return -(x - 1)


# Create the KDE plot for each distribution
fig, ax = plt.subplots(figsize=(3.5, 2))
sns.kdeplot(
    slopes_true,
    label=f"True ({slopes_true.mean():.3f})",
    color="grey",
    lw=2,
    ax=ax,
)
sns.kdeplot(
    slopes_corrected,
    label=f"Corrected ({slopes_corrected.mean():.3f})",
    color="#1b9e77",
    linestyle="dotted",
    ax=ax,
)
sns.kdeplot(
    slopes_naive,
    label=f"Naive ({slopes_naive.mean():.3f})*",
    color="indianred",
    linestyle="dashed",
    ax=ax,
)
sns.kdeplot(
    slopes_lint,
    label=f"LINT ({slopes_lint.mean():.3f})*",
    color="black",
    linestyle="dashdot",
    ax=ax,
)

# Add labels and title
plt.xlabel("Slope of fitted regression line to log-log SF")

# Create a secondary x-axis that is the negative of the original x-axis
secax = ax.secondary_xaxis("top", functions=(sf_to_psd, psd_to_sf))
secax.set_xlabel("Equivalent spectral index")

# Add vertical lines at the means of the distributions
plt.axvline(slopes_naive.mean(), color="indianred", linestyle="dashed", lw=0.7)
plt.axvline(slopes_lint.mean(), color="black", linestyle="dashdot", lw=0.7)
plt.axvline(slopes_corrected.mean(), color="#1b9e77", linestyle="dotted", lw=0.7)
plt.axvline(slopes_true.mean(), color="grey", linestyle="solid", lw=0.7)

# Add vertical line at 2/3
plt.axvline(2 / 3, color="mediumblue", linestyle="solid", alpha=0.3, lw=0.5)
# Annotate next to this line
plt.text(
    2 / 3 + 0.01,
    5,
    "K41 prediction",
    va="center",
    ha="left",
    fontsize=9,
    color="mediumblue",
    alpha=0.5,
)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
plt.yticks([])
plt.semilogy()
plt.ylim(1e-2, 1e1)
plt.xlim(-0.15, 1.25)
plt.savefig("plots/results/final/slope_hist_log.pdf", bbox_inches="tight")
# plt.show()

# STATISTICAL TEST OF DIFFERENCE OF MEANS
# Perform an ANOVA test to see if the means are significantly different?

# First, testing assumptions of ANOVA

# 1. Homogeneity of variances
print("\nResults of Levene's test for homogeneity of variances:")
print(levene(slopes_naive, slopes_lint, slopes_corrected, slopes_true))
# If the p-value is less than 0.05, then the variances are significantly different
print("HOMOGENEITY OF VARIANCES NOT SATISFIED")

# 2. Normality
print("\nResults of Shapiro-Wilk test for normality:")
print(shapiro(slopes_naive))
print(shapiro(slopes_lint))
print(shapiro(slopes_corrected))
print(shapiro(slopes_true))
print("NORMALITY NOT SATISFIED")
# If the p-value is less than 0.05, then the data is not normally distributed
print("\nCANNOT PROCEED WITH ANOVA")

# Combine all the data into one array
all_data = pd.DataFrame(
    {
        "slope": pd.concat([slopes_naive, slopes_lint, slopes_corrected, slopes_true]),
        "group": ["naive"] * len(slopes_naive)
        + ["lint"] * len(slopes_lint)
        + ["corrected"] * len(slopes_corrected)
        + ["orig"] * len(slopes_true),
    }
)

# Since we can't perform ANOVA, instead use non-parametric test for difference of means
# H0: The means of the groups are equal
# H1: At least one of the means is different
print("\nResults of non-parametric Kruskal-Wallis test for difference of means:")
print(kruskal(slopes_naive, slopes_lint, slopes_corrected, slopes_true))
print("REJECT NULL HYPOTHESIS OF EQUAL MEANS")

# If the Kruskal-Wallis test is significant, then perform a post-hoc test to see which groups are different
print("\nResults of post-hoc Dunn's test for difference of means:")
print(
    posthoc_dunn(
        all_data,
        val_col="slope",
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


# # Create histograms for all the different slopes
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
# plt.xlabel("Slope of fitted regression line to log-log SF")
# plt.ylabel("Density")
# plt.legend()
# plt.semilogy()
# # plt.ylim(1e-2, 1e1)
# plt.show()
