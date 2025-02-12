# This script performs statistical tests,
# comparing the distributions of the slopes, correlation scales, etc., for the different gap handling methods

# For publication, perhaps add asterisks to indicate significance of difference of means

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import wilcoxon

import src.params as params

run_mode = params.run_mode
spacecraft = "wind"
n_bins = 25


# Function to calculate effect sizes
def compute_effect_sizes(ref, sample):
    pe_mean = (
        (np.mean(sample) - np.mean(ref)) / np.abs(np.mean(ref)) * 100
    )  # Mean difference
    pe_median = (
        (np.median(sample) - np.median(ref)) / np.abs(np.median(ref))
    ) * 100  # Median difference
    pe_var = (np.var(sample, ddof=1) - np.var(ref, ddof=1)) / np.var(
        ref, ddof=1
    )  # Variance ratio
    ks_stat, ks_p = stats.ks_2samp(ref, sample)  # KS test
    anderson_p = stats.anderson_ksamp(
        [ref, sample],
        method=stats.PermutationMethod(),  # for dealing with capped/floored p-values
    ).pvalue  # Anderson-Darling test
    levene_stat, levene_p = stats.levene(ref, sample)  # Variance test
    fligner_stat, fligner_p = stats.fligner(ref, sample)  # Robust variance test
    mw_stat, mw_p = stats.mannwhitneyu(
        ref, sample, alternative="two-sided"
    )  # Location test
    wasserstein_dist = stats.wasserstein_distance(ref, sample)  # Wasserstein distance

    return {
        "Mean Diff (%)": pe_mean,
        "Median Diff (%)": pe_median,
        "Variance Diff (%)": pe_var,
        "Wasserstein Distance": wasserstein_dist,
        "Mann-Whitney p-value": mw_p,
        # "Levene p-value": levene_p,
        "Fligner p-value": fligner_p,
        "KS p-value": ks_p,
        "Anderson-Darling p-value": anderson_p,
    }


# Load data
ints = pd.read_csv(
    f"results/{run_mode}/test_{spacecraft}_corrected_{n_bins}_bins_ints_gapped_metadata.csv"
)
run_mode = params.run_mode

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

df_results_full = pd.DataFrame()

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

        true = data_true[variable + "_orig"]  # Reference distribution
        naive = data_naive[variable]
        corrected = data_corrected[variable]
        lint = data_lint[variable]

        # Perform comparisons
        comparison_results = [
            compute_effect_sizes(true, naive),
            compute_effect_sizes(true, corrected),
            compute_effect_sizes(true, lint),
        ]

        # Create and print results DataFrame
        df_results = pd.DataFrame(comparison_results)
        df_results["method"] = ["naive", "corrected", "lint"]
        df_results["variable"] = variable
        df_results["bin"] = bin

        df_results_full = pd.concat([df_results_full, df_results])

print("\nResults of statistical tests:")
print(df_results_full)


# # Assuming df is your DataFrame
# def plot_changes(df):
#     # numeric_vars = df.select_dtypes(include=["number"]).columns
#     var = "Mean Diff"
#     plt.figure(figsize=(10, 6))
#     for variable in variables:
#         sns.lineplot(
#             data=df[df["variable"] == variable],
#             x="bin",
#             y=var,
#             hue="method",
#             style="variable",
#             markers=True,
#         )
#     plt.title(f"Change in {var} Across Bins")
#     plt.xlabel("Bin")
#     plt.ylabel(var)
#     plt.xticks(rotation=45)
#     plt.legend(title="Method & Variable")
#     plt.grid(True)
#     plt.show()


# plot_changes(df_results_full)


df_to_plot = df_results_full[df_results_full["bin"] != "all_data"]


var_names = [
    r"$\beta$",
    r"$\lambda_C$ (lags)",
    r"$\lambda_T$ (lags)",
    r"$Re_{\lambda_T}$",
]

# Get count of numeric vars in df_to_plot
df_to_plot.head()

# Get list of numerical columns in df_to_plot
metrics = df_to_plot.columns[df_to_plot.dtypes == "float64"]
metrics.values


fig, ax = plt.subplots(len(metrics), len(variables), figsize=(10, 2 * len(metrics)))

for i, metric in enumerate(metrics):
    for j, variable in enumerate(variables):
        sns.lineplot(
            data=df_to_plot[df_to_plot["variable"] == variable],
            x="bin",
            y=metric,
            hue="method",
            palette=["indianred", "#1b9e77", "black"],
            markers=True,
            style="method",
            dashes=False,
            ax=ax[i, j],
        )
        if i == 0:
            ax[i, j].set_title(var_names[j], fontsize=14)
        if j == 0:
            ax[i, j].set_ylabel(metric)
        else:
            ax[i, j].set_ylabel("")

        if i < 4:
            ax[i, j].axhline(0, color="black", linestyle="--", alpha=0.5)
        else:
            ax[i, j].set_ylim(0, 1)
        ax[i, j].set_xlabel("")
        ax[i, j].grid(True)

# Create a shared legend
handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(labels),
    bbox_to_anchor=(0.5, 0.985),
    fontsize=12,
)

# Remove individual legends
for i in range(len(metrics)):
    for j in range(len(variables)):
        ax[i, j].get_legend().remove()

# Reduce space between subplots
plt.tight_layout(h_pad=1.2, w_pad=0.2, rect=[0, 0, 1, 0.96])
# Add a horizontal line between the 4th and 5th rows
line_y = 0.475  # Normalized figure coordinates (0=bottom, 1=top)
fig.add_artist(
    plt.Line2D(
        [0, 1], [line_y, line_y], color="black", linewidth=2, transform=fig.transFigure
    )
)

plt.suptitle(
    "Quantifying Change in PDFs of SF-Derived Stats With Increasing Missing Data",
    fontsize=18,
    y=1,
)
# plt.savefig("all_metrics_pe_var.png")
plt.show()
sys.exit()
# My changes
# - Make all_data a horizontal line
# - A column for each variable


# above: code to get test results
# below: code to plot histograms of slopes with p-values

# what I want: to test new tests with slopes,
# add p-values to the histograms,
# and then extend to other variables above


# Switch the rows and columns
df_results.T


# Naive has marginally better center
## closer mean and median, latter of which results in non-significant Mann-Whitney p-value
## variance ratio over 2, significant Fligner p-values
## higher Wasserstein

# Corrected has much more accurate spread
## Larger mean and median, resulting in significant Mann-Whitney p-value
## Variance ratio close to 1, Fligner p-value not significant
## Wasserstein distance is lower

# Both have significant KS values at 99.9% confidence


# List of groups to compare with Group A
groups = [("Naive", data_naive), ("Corrected", data_corrected), ("Lint", data_lint)]

# Set consistent bins
_, bins, _ = plt.hist(
    data_naive["es_slope"],
    bins=50,
)

# Create subplots for each pairwise comparison
fig, axes = plt.subplots(1, len(groups), figsize=(12, 3), sharey=True, sharex=True)

# for row, cumulative in enumerate([False, True]):
for ax, (group_name, group_data) in zip(axes, groups):
    ax.hist(
        data_true.es_slope_orig,
        bins=bins,
        alpha=0.5,
        label="True",
        color="blue",
        cumulative=False,
    )
    ax.hist(
        group_data["es_slope"],
        bins=bins,
        alpha=0.5,
        label=group_name,
        color="orange",
        cumulative=False,
    )
    ax.set_title(f"True vs. {group_name}")
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequency")
    ax.legend()
    # Add a line at the median of each group
    ax.axvline(data_true.es_slope_orig.median(), color="blue", linestyle="--")
    ax.axvline(group_data["es_slope"].median(), color="orange", linestyle="--")

    ax.axvline(data_true.es_slope_orig.min(), color="blue", alpha=0.5, linestyle=":")
    ax.axvline(data_true.es_slope_orig.max(), color="blue", alpha=0.5, linestyle=":")
    ax.axvline(group_data["es_slope"].min(), color="orange", alpha=0.5, linestyle=":")
    ax.axvline(group_data["es_slope"].max(), color="orange", alpha=0.5, linestyle=":")

    # # Add annotation of Dunn's test results
    # p_value = posthoc_dunn(
    #     all_data, val_col=variable, group_col="group", p_adjust="bonferroni"
    # ).loc["orig", f"{group_name.lower()}"]
    # ax.text(
    #     0.05,
    #     0.8,
    #     f"p={p_value:.2f}",
    #     horizontalalignment="left",
    #     verticalalignment="center",
    #     transform=ax.transAxes,
    # )
    # Print sample size
    ax.text(
        0.05,
        0.7,
        f"n={len(group_data)}",
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax.transAxes,
    )

# Adjust layout for better visualization
plt.suptitle(f"Comparison of True vs. Naive, Corrected, Lint for {variable}, bin {bin}")
plt.tight_layout()
plt.show()
# plt.savefig(f"{variable}_dists_{bin}.png")

print("FINISHED")


# Multiple test correction just doubles the p-values as we are conducting 2 tests

# # Extract p-values for each test separately
# ks_pvalues = [res["KS p-value"] for res in comparison_results]
# levene_pvalues = [res["Levene p-value"] for res in comparison_results]
# fligner_pvalues = [res["Fligner p-value"] for res in comparison_results]
# mw_pvalues = [res["Mann-Whitney p-value"] for res in comparison_results]

# # Apply Bonferroni correction to each set of p-values
# # (return first value instead for reject decision)
# _, corrected_ks_pvals, *_ = multipletests(ks_pvalues, method="bonferroni")
# _, corrected_levene_pvals, *_ = multipletests(levene_pvalues, method="bonferroni")
# _, corrected_fligner_pvals, *_ = multipletests(fligner_pvalues, method="bonferroni")
# _, corrected_mw_pvals, *_ = multipletests(mw_pvalues, method="bonferroni")

# # Assign corrected p-values to the results
# for i, res in enumerate(comparison_results):
#     res["Corrected KS p-value"] = corrected_ks_pvals[i]
#     res["Corrected Levene p-value"] = corrected_levene_pvals[i]
#     res["Corrected Fligner p-value"] = corrected_fligner_pvals[i]
#     res["Corrected Mann-Whitney p-value"] = corrected_mw_pvals[i]
