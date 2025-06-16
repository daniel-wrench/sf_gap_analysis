import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set matplotlib styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in"})

# Read and process data
df_v1 = pd.read_csv("results/full/voyager1_corrected_metadata_NEW_RANGE.csv")
high_res_stats_V1 = pd.read_csv("results/full/voyager1_lism_ttu_hr.csv")
high_res_stats_V1["source"] = "Voyager 1"

# Calculate time columns
df_v1["tce_s"] = df_v1["tce"] * df_v1["cadence"]
df_v1["ttu_s"] = df_v1["ttu"] * df_v1["cadence"]
df_v1["tce_days"] = df_v1["tce_s"] / (24 * 3600)
df_v1["ttu_hours"] = df_v1["ttu_s"] / 3600
df_v1["source"] = "Voyager 1"

# Read and process data
df_v2 = pd.read_csv("results/full/voyager2_corrected_metadata_NEW_RANGE.csv")

# Calculate time columns
df_v2["tce_s"] = df_v2["tce"] * df_v2["cadence"]
df_v2["ttu_s"] = df_v2["ttu"] * df_v2["cadence"]
df_v2["tce_days"] = df_v2["tce_s"] / (24 * 3600)
df_v2["ttu_hours"] = df_v2["ttu_s"] / 3600
df_v2["source"] = "Voyager 2"

df = pd.concat([df_v1, df_v2], ignore_index=True)

# Fraternale 2021 data
fraternale_data = pd.DataFrame(
    {
        "tce_days": [63.5, 19, 76, 19, 60, np.nan, np.nan, np.nan],
        "ttu_hours": [0.048, 0.06, 0.07, 0.054, 0.063, np.nan, np.nan, np.nan],
        "slope": [0.36, 0.28, 0.27, 0.29, 0.67, 0.48, 0.37, 0.72],
        "source": ["Previous V1 estimates"] * 8,
    }
)

# Plot setup
fig, axes = plt.subplots(1, 3, figsize=(5, 2), sharey=True)
axes = axes.flatten()

vars_to_plot = ["slope", "tce_days", "ttu_hours"]
labels = [r"$\beta$", r"$\lambda_C$ (days)", r"$\lambda_T$ (hours)"]

bin_edges = {
    "slope": np.linspace(0.1, 0.9, 10),
    "tce_days": np.linspace(0, 70, 10),
    "ttu_hours": np.linspace(0, 0.1, 10),
}

# Store handles and labels for shared legend
legend_handles = None
legend_labels = None

for i, var in enumerate(vars_to_plot):
    # Main data
    if var == "ttu_hours":
        current_data = high_res_stats_V1
    else:
        current_data = df[df["gap_handling"] == "corrected_3d"]
    current_data = pd.concat([current_data, fraternale_data], ignore_index=True)

    # Plot with legend only on first axis to grab handles
    g = hist = sns.histplot(
        current_data,
        x=var,
        hue="source",
        hue_order=[
            "Voyager 1",
            "Voyager 2",
            "Previous V1 estimates",
        ],
        palette=["#2ca02c", "#1f77b4", "#ff7f0e"],
        ax=axes[i],
        bins=bin_edges[var],
        element="step",
        legend=(i == 1),  # ✅ Enable legend only for the first plot
    )

    axes[i].set_xlim(bin_edges[var][0], bin_edges[var][-1])
    axes[i].set_xlabel(labels[i], fontsize=10)

    if var == "slope":
        axes[i].axvline(2 / 3, color="black", linestyle="dotted", alpha=0.6)
        axes[i].text(2 / 3 + 0.01, 6, "K41", fontsize=10, alpha=0.6)

    print(
        (
            f"95% CI for {var}: "
            f"{current_data[var].quantile(0.025):.2f}, "
            f"{current_data[var].quantile(0.975):.2f}, "
            f"median: {current_data[var].median():.2f}"
        )
    )
sns.move_legend(axes[1], "upper center", bbox_to_anchor=(0.5, 1.4), ncol=3, title="")

axes[0].set_ylabel("Count", fontsize=10)
max_y_val = max([ax.get_ylim()[1] for ax in axes])
for ax in axes:
    ax.set_ylim(0, max_y_val)

# Adjust layout to make room for legend
plt.tight_layout()
plt.subplots_adjust(top=0.8, wspace=0.1)
plt.savefig(
    "results/full/plots/voyager/voyager1_2_corrected_stats_NEW_RANGE.png",
    dpi=300,
    bbox_inches="tight",
)
print("Done")
