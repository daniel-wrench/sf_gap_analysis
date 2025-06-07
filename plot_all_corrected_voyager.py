import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Replace with your pickle file path
pickle_file = "results/full/voyager2_corrected_sfs.pkl"

# Load the list of time series
with open(pickle_file, "rb") as f:
    sfs_gapped_corrected_all = pickle.load(f)


sfs_corrected = sfs_gapped_corrected_all[
    sfs_gapped_corrected_all.gap_handling == "corrected_3d"
]
sfs_corrected["distance"] = sfs_corrected["int_index"] * 1.67 + 120
sfs_corrected["lag_s"] = sfs_corrected["lag"] * 288  # Convert to seconds
sfs_corrected["inverse_lag_s"] = 1 / sfs_corrected["lag_s"]
df = sfs_corrected.copy()

# Get unique distances for color mapping
distances = df["distance"].unique()

# Create colormap
cmap = plt.cm.plasma  # You can change this to any colormap
norm = plt.Normalize(vmin=distances.min(), vmax=distances.max())

# Plot each time series
time_col = "lag_s"
value_col = "sf_2"

fig, ax = plt.subplots(figsize=(5, 3))

for distance in distances:
    series_data = df[df["distance"] == distance]
    color = cmap(norm(distance))
    ax.loglog(
        series_data[time_col],
        series_data[value_col],
        color=color,
        alpha=0.7,
        linewidth=1,
    )

# Plot 2/3 power law
ax.loglog(
    df[time_col],
    (df[time_col] ** (2 / 3)) * 0.01,
    color="black",
    linestyle="--",
    linewidth=1,
    alpha=0.5,
    label="$\\tau^{2/3}$",
)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Distance (au)", rotation=270, labelpad=20)
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))

ax.set_xlabel("$\\tau$ (s)")
ax.set_ylabel("$S_2$")
ax.set_title("All Corrected Voyager 2 SFs")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()

plt.savefig(
    "results/full/plots/voyager/voyager2_corrected_sfs.png",
    dpi=300,
    bbox_inches="tight",
)

# EQUIVALENT SPECTRA

# Plot each time series
time_col = "inverse_lag_s"
value_col = "sf_corrected_es"

fig, ax = plt.subplots(figsize=(5, 3))

for distance in distances:
    series_data = df[df["distance"] == distance]
    color = cmap(norm(distance))
    ax.loglog(
        series_data[time_col],
        series_data[value_col],
        color=color,
        alpha=0.7,
        linewidth=1,
    )

# Plot 2/3 power law
ax.loglog(
    df[time_col],
    (df[time_col] ** (-5 / 3)) * 0.00001,
    color="black",
    linestyle="--",
    linewidth=1,
    alpha=0.5,
    label="f$^{-5/3}$",
)

ax.loglog(
    df[time_col],
    (df[time_col] ** (-1)) * 0.01,
    color="black",
    linestyle=":",
    linewidth=1,
    alpha=0.5,
    label="f$^{-1}$",
)


# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Distance (au)", rotation=270, labelpad=20)
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("ES")
ax.set_title("All Corrected Voyager 2 ES")
ax.grid(True, alpha=0.3)

ax.legend()
plt.tight_layout()
plt.savefig(
    "results/full/plots/voyager/voyager2_corrected_es.png",
    dpi=300,
    bbox_inches="tight",
)
