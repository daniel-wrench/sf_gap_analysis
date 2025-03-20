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
df.loc[:, "ttu_hours"] = df["ttu_s"] / (3600)

df.loc[:, "source"] = "current work"

# Add new rows from lit review
new_rows = pd.DataFrame(
    {
        "tce_days": [63.5, 19, 76, 19, 60],
        "ttu_hours": [0.048, 0.06, 0.07, 0.054, 0.063],
        "source": ["Frat2021", "Frat2021", "Frat2021", "Frat2021", "Frat2021"],
    }
)

df = pd.concat([df, new_rows], ignore_index=True)
# Create a figure with 6 subplots (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
axes = axes.flatten()  # Flatten to make indexing easier

# Plot histogram of missing data
vars_to_plot = ["slope", "es_pwr_law_slope", "missing", "tce_days", "ttu_hours"]

for i, var in enumerate(vars_to_plot):
    sns.histplot(df, x=var, hue="source", multiple="stack", ax=axes[i], bins=10)
    # Add vertical line at 17 to axes 3

    axes[i].set_xlabel(var, labelpad=-10, fontsize=12)  # Adjust x-axis label
    axes[i].xaxis.set_label_position("top")  # Move x-axis label to top
    axes[i].set_ylabel("")  # Remove y-axis label
    # Move legend

    if i == 3:
        axes[i].axvline(
            17, color="black", linestyle="--", label=r"Ostensible $\\lambda_C$"
        )
        sns.move_legend(axes[i], "center right")

    else:
        axes[i].get_legend().remove()  # Remove legend from all but last panel

fig.delaxes(axes[-1])  # Remove empty subplot

plt.tight_layout()

plt.suptitle(
    "Distribution of stats from 24 corrected V1 LISM structure functions",
    fontsize=16,
)
plt.subplots_adjust(top=0.9)  # Make room for the title

# Save or display the plot
plt.savefig(
    "results/full/plots/voyager/voyager_corrected_stats.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
