# SLOPE HISTOGRAM PLOT

# Get unique vals of slope_orig column
# Add gap_handling = original column
# Set slope_orig = slope
# Append to dataframe
# Get counts of each gap_handling
# Create overlapping histograms of each gap handling method, in colours of other keys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# Load data
ints = pd.read_csv("gapped_ints_metadata.csv")

slopes_naive = ints[ints.gap_handling == "naive"].slope
slopes_lint = ints[ints.gap_handling == "lint"].slope
slopes_corrected = ints[ints.gap_handling == "corrected_3d"].slope

slopes_orig = ints[ints.gap_handling == "naive"].slope_orig

# Create the KDE plot for each distribution
plt.figure(figsize=(5, 2.5))
sns.kdeplot(
    slopes_orig,
    label=f"Original ({slopes_orig.mean():.3f})",
    fill=False,
    color="grey",
    linewidth=2,
    alpha=0.5,
)
sns.kdeplot(
    slopes_corrected,
    label=f"Corrected ({slopes_corrected.mean():.3f})",
    fill=False,
    color="#1b9e77",
    linestyle="-.",
)
sns.kdeplot(
    slopes_naive,
    label=f"Naive ({slopes_naive.mean():.3f})",
    fill=False,
    color="indianred",
    linestyle="--",
)
sns.kdeplot(
    slopes_lint,
    label=f"LINT ({slopes_lint.mean():.3f})",
    fill=False,
    color="black",
    linestyle=":",
)


# Add labels and title
plt.xlabel("Fitted slope in inertial range")

plt.legend()
plt.yticks([])
plt.savefig("plots/results/final/slope_hist.pdf", bbox_inches="tight")
