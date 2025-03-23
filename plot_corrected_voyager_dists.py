import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set matplotlib styling
plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in"})

# Read and process data
df = pd.read_csv("results/full/voyager1_corrected_metadata.csv")

# Calculate time columns
df["tce_s"] = df["tce"] * df["cadence"]
df["ttu_s"] = df["ttu"] * df["cadence"]
df["tce_days"] = df["tce_s"] / (24 * 3600)
df["ttu_hours"] = df["ttu_s"] / 3600
df["source"] = "current work"

# Add Fraternale 2021 data
fraternale_data = pd.DataFrame(
    {
        "tce_days": [63.5, 19, 76, 19, 60],
        "ttu_hours": [0.048, 0.06, 0.07, 0.054, 0.063],
        "source": ["Frat2021"] * 5,
    }
)

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
axes = axes.flatten()

# Variables to plot
vars_to_plot = ["slope", "es_pwr_law_slope", "missing", "tce_days", "ttu_hours"]
colors = {"current work": "steelblue", "Frat2021": "crimson"}

# Plot histograms and marks
for i, var in enumerate(vars_to_plot):
    # Plot histogram for current work data only
    current_data = df[df["source"] == "current work"]
    sns.histplot(
        current_data,
        x=var,
        color=colors["current work"],
        ax=axes[i],
        bins=10,
        label="current work",
    )

    # Add Fraternale data as crosses on x-axis if the variable exists in fraternale_data
    if var in fraternale_data.columns:
        frat_values = fraternale_data[var].dropna()
        if len(frat_values) > 0:
            y_pos = np.zeros_like(frat_values)
            axes[i].scatter(
                frat_values,
                y_pos + 0.3,
                marker="*",
                s=80,
                color=colors["Frat2021"],
                label="Frat2021",
                zorder=5,
            )

    # Styling
    axes[i].set_xlabel(var, labelpad=-10, fontsize=12)
    axes[i].xaxis.set_label_position("top")
    axes[i].set_ylabel("")

    # Handle special case for tce_days
    if var == "tce_days":
        axes[i].axvline(
            17, color="black", linestyle="--", label=r"Ostensible $\lambda_C$"
        )
        handles, labels = axes[i].get_legend_handles_labels()
        if len(handles) < 2:  # If Frat2021 not in this plot
            # Add dummy Frat2021 entry for legend
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    markerfacecolor=colors["Frat2021"],
                    markersize=8,
                    markeredgecolor=colors["Frat2021"],
                )
            )
            labels.append("Frat2021")
        axes[i].legend(handles, labels, loc="center right")
    else:
        # Remove automatically generated legends
        if axes[i].get_legend():
            axes[i].get_legend().remove()

# Remove the empty subplot
fig.delaxes(axes[-1])

# Finalize layout
plt.tight_layout()
plt.suptitle(
    "Distribution of stats from 24 corrected V1 LISM structure functions", fontsize=16
)
plt.subplots_adjust(top=0.9)  # Make room for the title

# Save and display
# plt.savefig(
#     "results/full/plots/voyager/voyager_corrected_stats.png",
#     dpi=300,
#     bbox_inches="tight",
# )
plt.show()
