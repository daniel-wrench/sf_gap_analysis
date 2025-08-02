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
df = pd.read_csv("results/full/voyager1_corrected_metadata_NEW.csv")
high_res_stats = pd.read_csv("results/full/voyager1_lism_ttu_hr.csv")

# Calculate time columns
df["tce_s"] = df["tce"] * df["cadence"]
df["ttu_s"] = df["ttu"] * df["cadence"]
df["tce_days"] = df["tce_s"] / (24 * 3600)
df["ttu_hours"] = df["ttu_s"] / 3600

# Fraternale 2021 data
fraternale_data = pd.DataFrame(
    {
        "tce_days": [63.5, 19, 76, 19, 60, np.nan, np.nan, np.nan],
        "ttu_hours": [0.048, 0.06, 0.07, 0.054, 0.063, np.nan, np.nan, np.nan],
        "slope": [0.36, 0.28, 0.27, 0.29, 0.67, 0.48, 0.37, 0.72],
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

frat_color = "skyblue"
main_color = {"slope": "black", "tce_days": "black", "ttu_hours": "indianred"}

for i, var in enumerate(vars_to_plot):
    # Main data
    if var == "ttu_hours":
        current_data = high_res_stats
    else:
        current_data = df[df["gap_handling"] == "corrected_3d"]

    # Prepare data for side-by-side bars
    main_values = current_data[var].dropna()
    frat_values = (
        fraternale_data[var].dropna()
        if var in fraternale_data.columns
        else pd.Series(dtype=float)
    )
    bins = bin_edges[var]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.4  # 40% of bin width for each bar

    main_hist, _ = np.histogram(main_values, bins=bins)
    frat_hist, _ = np.histogram(frat_values, bins=bins)

    # Plot main bars
    axes[i].bar(
        bin_centers - width / 2,
        main_hist,
        width=width,
        color=main_color[var],
        alpha=0.7,
        label="This work",
        align="center",
    )
    # Plot Fraternale bars
    axes[i].bar(
        bin_centers + width / 2,
        frat_hist,
        width=width,
        color=frat_color,
        alpha=0.7,
        label="Fraternale et al.",
        align="center",
        edgecolor="black",
        linewidth=1,
    )

    axes[i].set_xlim(bin_edges[var][0], bin_edges[var][-1])
    axes[i].set_xlabel(labels[i], fontsize=10)

    if var == "slope":
        axes[i].axvline(2 / 3, color="black", linestyle="dotted", alpha=0.6)
        axes[i].text(2 / 3 + 0.01, 6, "K41", fontsize=10, alpha=0.6)

    print(
        f"95% CI for {var}: {main_values.quantile(0.025):.2f}, {main_values.quantile(0.975):.2f}, median: {main_values.median():.2f}"
    )

axes[0].set_ylabel("Count", fontsize=10)
max_y_val = max([ax.get_ylim()[1] for ax in axes])
for ax in axes:
    ax.set_ylim(0, max_y_val)

plt.tight_layout()
axes[0].legend(fontsize=8, frameon=False)
plt.subplots_adjust(top=0.8, wspace=0.1)
plt.show()
print("Done")
