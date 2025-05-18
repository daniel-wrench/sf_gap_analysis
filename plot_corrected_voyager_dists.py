import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set matplotlib styling
# plt.rc("text", usetex=True)
# plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in"})

# Read and process data
df = pd.read_csv("results/full/voyager1_corrected_metadata_NEW.csv")

# Read in high-res stats
high_res_stats = pd.read_csv("results/full/voyager1_lism_ttu_hr.csv")

# Calculate time columns
df["tce_s"] = df["tce"] * df["cadence"]
df["ttu_s"] = df["ttu"] * df["cadence"]
df["tce_days"] = df["tce_s"] / (24 * 3600)
df["ttu_hours"] = df["ttu_s"] / 3600

# Add Fraternale 2021 data
fraternale_data = pd.DataFrame(
    {
        "tce_days": [63.5, 19, 76, 19, 60, np.nan, np.nan, np.nan],  # days
        "ttu_hours": [0.048, 0.06, 0.07, 0.054, 0.063, np.nan, np.nan, np.nan],
        "slope": [0.36, 0.28, 0.27, 0.29, 0.67, 0.48, 0.37, 0.72],  # mins  # maxs
    }
)

########

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(5, 2), sharey=True)
axes = axes.flatten()

# Variables to plot
vars_to_plot = ["slope", "tce_days", "ttu_hours"]
labels = [r"$\beta$", r"$\lambda_C$ (days)", r"$\lambda_T$ (hours)"]

# Define explicit bin edges for each variable to ensure consistent visual width
bin_edges = {
    "slope": np.linspace(0.1, 0.9, 10),  # 8 bins of width 0.1
    "tce_days": np.linspace(0, 70, 10),  # 7 bins of width 10
    "ttu_hours": np.linspace(0, 0.1, 10),  # 7 bins of width 0.1
}

# Plot histograms and marks
for i, var in enumerate(vars_to_plot):
    # Plot histogram for current work data only
    if var == "ttu_hours":
        current_data = high_res_stats
        sns.histplot(
            current_data,
            x=var,
            color="indianred",
            ax=axes[i],
            bins=bin_edges[var],
            alpha=0.9,
            element="bars",
        )
    else:
        current_data = df[df["gap_handling"] == "corrected_3d"]
        sns.histplot(
            current_data,
            x=var,
            color="black",
            ax=axes[i],
            bins=bin_edges[var],
            alpha=0.5,
            element="bars",
        )

    # Add Fraternale data as stars on x-axis if the variable exists in fraternale_data
    if var in fraternale_data.columns:
        frat_values = fraternale_data[var].dropna()
        if len(frat_values) > 0:
            y_pos = np.zeros_like(frat_values)
            axes[i].scatter(
                frat_values,
                y_pos + 0.8,
                marker="*",
                edgecolors="black",
                linewidths=0.5,
                alpha=0.7,
                s=100,
                color="skyblue",
                label="Frat2021",
                zorder=5,
            )

    # Set x-axis limits to match the bin edges
    axes[i].set_xlim(bin_edges[var][0], bin_edges[var][-1])

    # Styling
    axes[i].set_xlabel(labels[i], fontsize=10)
    # axes[i].xaxis.set_label_position("top")

    # Additional elements based on variable
    if var == "slope":
        axes[i].axvline(2 / 3, color="black", linestyle="dotted", alpha=0.6)
        axes[i].text(2 / 3 + 0.01, 6, "K41", fontsize=10, alpha=0.6)

    # Print the 95% confidence interval
    print(
        f"95% CI for {var}: {current_data[var].quantile(0.025):.2f}, {current_data[var].quantile(0.975):.2f}, median: {current_data[var].median():.2f}"
    )

    # elif var == "ttu_hours":
    # axes[i].text(
    #     0.95,
    #     0.8,
    #     "(Derived from Naive SF,\ninstead of corrected)",
    #     transform=axes[i].transAxes,
    #     fontsize=8,
    #     verticalalignment="top",
    #     horizontalalignment="right",
    #     bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    #     color="red",
    # )

# Set common y-label and y-limit
axes[0].set_ylabel("Count", fontsize=10)
max_y_val = max([ax.get_ylim()[1] for ax in axes])
for ax in axes:
    ax.set_ylim(0, max_y_val)

# Finalize layout
plt.tight_layout()
# plt.suptitle(
#     "SF-Derived Statistics from Voyager 1 Intervals of the Interstellar Medium",
#     fontsize=14,
#     y=0.98,
# )

# Create a legend entry matching the plotted stars
frat_legend = mlines.Line2D(
    [],
    [],
    color="skyblue",
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
    bbox_to_anchor=(0.5, 0.95),
    fontsize=8,
    frameon=False,
)

plt.subplots_adjust(top=0.8, wspace=0.1)  # Make room for the title and adjust spacing

plt.savefig(
    "results/full/plots/voyager/voyager_corrected_stats_NEW_TTU.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
print("Done")
