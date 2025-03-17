# Read in results/full/voyager1_corrected_metadata.csv

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set matplotlib font size
plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# Read in the data
df = pd.read_csv("results/full/voyager1_corrected_metadata.csv")

df.loc[:, "tce_s"] = df["tce"] * df["cadence"]
df.loc[:, "ttu_s"] = df["ttu"] * df["cadence"]

df.loc[:, "tce_days"] = df["tce_s"] / (24 * 3600)
df.loc[:, "ttu_days"] = df["ttu_s"] / (24 * 3600)

# Create a figure with 5 subplots (1 row, 5 columns)
fig, axes = plt.subplots(2, 3, figsize=(8, 4))
axes = axes.flatten()  # Flatten to make indexing easier

# Plot histogram of missing data
vars_to_plot = ["slope", "es_pwr_law_slope", "missing", "tce_days", "ttu_days"]

for i, var in enumerate(vars_to_plot):
    axes[i].hist(df[var])
    # sns.histplot(df[var], ax=axes[i])
    axes[i].set_xlabel(var)

# Remove the unused 6th subplot
fig.delaxes(axes[5])

# Adjust layout
plt.tight_layout()

plt.suptitle(
    "Distribution of stats from corrected V1 LISM structure functions",
    fontsize=16,  # , y=1.02
)
plt.subplots_adjust(top=0.9)  # Make room for the title

# Save or display the plot
plt.savefig("results/full/voyager_corrected_stats.png", dpi=300, bbox_inches="tight")
plt.show()
