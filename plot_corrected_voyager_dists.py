import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set matplotlib styling
plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in"})

# Read and process data
df = pd.read_csv("results/full/voyager1_corrected_metadata_NEW.csv")

# Calculate time columns
df["tce_s"] = df["tce"] * df["cadence"]
df["ttu_s"] = df["ttu"] * df["cadence"]
df["tce_days"] = df["tce_s"] / (24 * 3600)
df["ttu_hours"] = df["ttu_s"] / 3600
# df["source"] = "current work"

# Add Fraternale 2021 data
fraternale_data = pd.DataFrame(
    {
        "tce_days": [63.5, 19, 76, 19, 60, np.nan, np.nan, np.nan],  # days
        "ttu_hours": [0.048, 0.06, 0.07, 0.054, 0.063, np.nan, np.nan, np.nan],
        "slope": [0.36, 0.28, 0.27, 0.29, 0.67, 0.48, 0.37, 0.72],  # mins  # maxs
        "source": ["Frat2021"] * 8,
    }
)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(8, 2.5), sharey=True)
axes = axes.flatten()

# Variables to plot
vars_to_plot = ["slope", "tce_days", "ttu_hours"]
# colors = {"current work": "#7fc97f", "Frat2021": "#beaed4"}
# colors = {"naive":"red",  "lint":"black", "corrected_3d":"#1b9e77"}
# Plot histograms and marks
for i, var in enumerate(vars_to_plot):
    # Plot histogram for current work data only
    if var == "ttu_hours":
        current_data = df[df["gap_handling"] == "naive"]
        sns.histplot(
            current_data,
            x=var,
            color="red",
            ax=axes[i],
            # bins=10,
        )
    else:
        current_data = df[df["gap_handling"] == "corrected_3d"]
        sns.histplot(
            current_data,
            x=var,
            color="#1b9e77",
            ax=axes[i],
            # bins=10,
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
                # Add thin black outline
                edgecolors="black",
                linewidths=0.5,
                s=80,
                color="#beaed4",
                label="Frat2021",
                zorder=5,
            )

    # Styling
    # Format axis labels with proper units
    if var == "tce_days":
        axes[i].set_xlabel(r"$\lambda_C$ (days)", labelpad=-15, fontsize=12)
        # axes[i].axvline(17, color="black", linestyle="--", label=r"Assumed $\lambda_C$")
        # axes[i].text(17 + 0.01, 5, r"Ostensible $\lambda_C$", fontsize=10)
    elif var == "ttu_hours":
        axes[i].set_xlabel(r"$\lambda_T$ (hours)", labelpad=-15, fontsize=12)
        # Add a curved arrow with annotation
        axes[i].text(
            0.95,
            0.8,
            "(Derived from Naive SF,\ninstead of corrected)",
            transform=axes[i].transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            color="red",
        )
    else:
        axes[i].set_xlabel("Inertial range slope", labelpad=-15, fontsize=12)
        axes[i].axvline(2 / 3, color="black", linestyle="dotted", label="Kolmogorov")
        # Add annotation next to this line
        axes[i].text(2 / 3 + 0.01, 3, r"\textit{K41}", fontsize=10)
    axes[i].xaxis.set_label_position("top")
    axes[i].set_ylabel("")

# Finalize layout
plt.tight_layout()
plt.suptitle(
    "SF-Derived Statistics from Voyager 1 Intervals of the Interstellar Medium",
    fontsize=16,
)
import matplotlib.lines as mlines

# Create a legend entry matching the plotted stars
frat_legend = mlines.Line2D(
    [],
    [],
    color="#beaed4",
    marker="*",
    linestyle="None",
    markersize=10,
    markeredgecolor="black",
    markeredgewidth=0.5,
    label="Values from Fraternale et al. (2019, 2021)",
)

# Add the legend just below the title
fig.legend(
    handles=[frat_legend],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.92),
    fontsize=10,
    frameon=False,
)

plt.subplots_adjust(top=0.8)  # Make room for the title

# Save and display
plt.savefig(
    "results/full/plots/voyager/voyager_corrected_stats.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
print("Done")
