# This script performs statistical tests,
# comparing the distributions of the slopes, correlation scales, etc., for the different gap handling methods

# For publication, perhaps add asterisks to indicate significance of difference of means

import numpy as np
import pandas as pd
from scikit_posthocs import posthoc_dunn
from scipy.stats import kruskal, levene, shapiro, wilcoxon

import src.params as params

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
    f"Total of {n_combinations} intervals, gapped {len(ints.version.unique())} times each = {n_combinations * len(ints.version.unique())} total intervals"
)

# Define the variables to plot
variables = ["es_slope", "tce", "ttu", "Re_lt"]

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

    for i, variable in enumerate(variables):

        # STATISTICAL TEST OF DIFFERENCE OF MEANS
        # Perform an ANOVA test to see if the means are significantly different?
        print("\n###########################################")
        print(f"\nResults for {variable}, bin {bin} \nstarting with summary stats\n")

        print("\nTrue:\n")
        print(data_true[variable + "_orig"].describe())

        print("\nNaive:\n")
        print(data_naive[variable].describe())

        print("\nLINT:\n")
        print(data_lint[variable].describe())

        print("\nCorrected:\n")
        print(data_corrected[variable].describe())

        if variable == "es_slope":

            # Perform a Wilcoxon signed-rank test to see if the means are significantly different from -5/3
            print(
                "\n\nResults of Wilcoxon signed-rank test for difference of mean of True distribution from -5/3:"
            )
            print(wilcoxon(data_true[variable + "_orig"] + 5 / 3))
            print("(If the p-value < 0.05, then the means are significantly different)")

        # First, testing assumptions of ANOVA

        # 1. Homogeneity of variances
        print(
            f"\nResults of Levene's test for homogeneity of variances for {variable}:"
        )
        print("(If p-value < 0.05, then the variances are significantly different)\n")
        print(
            levene(
                data_naive[variable],
                data_lint[variable],
                data_corrected[variable],
                data_true[variable + "_orig"],
            )
        )

        # 2. Normality
        print(f"\nResults of Shapiro-Wilk test for normality for {variable}:")
        print("(If p-value < 0.05, then the data is not normally distributed)\n")
        print(shapiro(data_naive[variable]))
        print(shapiro(data_lint[variable]))
        print(shapiro(data_corrected[variable]))
        print(shapiro(data_true[variable + "_orig"]))

        # Combine all the data into one array
        all_data = pd.DataFrame(
            {
                variable: pd.concat(
                    [
                        data_naive[variable],
                        data_lint[variable],
                        data_corrected[variable],
                        data_true[variable + "_orig"],
                    ]
                ),
                "group": ["naive"] * len(data_naive[variable])
                + ["lint"] * len(data_lint[variable])
                + ["corrected"] * len(data_corrected[variable])
                + ["orig"] * len(data_true[variable + "_orig"]),
            }
        )

        # Since we can't perform ANOVA, instead use non-parametric test for difference of means
        # H0: The means of the groups are equal
        # H1: At least one of the means is different
        print(
            f"\nResults of non-parametric Kruskal-Wallis test for difference of means for {variable}:"
        )
        print("(If p-value < 0.05, then the means are significantly different)\n")
        print(
            kruskal(
                data_naive[variable],
                data_lint[variable],
                data_corrected[variable],
                data_true[variable + "_orig"],
            )
        )

        # If the Kruskal-Wallis test is significant, then perform a post-hoc test to see which groups are different
        print(
            f"\nResults of post-hoc Dunn's test (adjusted with Bonferroni) of pairwise comparisions of {variable}:"
        )
        print("(If p-value < 0.05, then the means are significantly different)\n")
        print(
            posthoc_dunn(
                all_data,
                val_col=variable,
                group_col="group",
                p_adjust="bonferroni",
            )
        )

print("FINISHED")
