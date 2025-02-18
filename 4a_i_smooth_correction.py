import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

import src.params as params

n_bins = 25
run_mode = params.run_mode
dim = 3

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


def smooth_with_missing_values(image, sigma):
    # 1. Create a mask of NaNs (True where NaNs are)
    nan_mask = np.isnan(image)

    # 2. Replace NaNs with a temporary value (e.g., the mean of the non-NaN values)
    image_filled = np.where(nan_mask, np.nanmean(image), image)

    # 3. Apply Gaussian filter to the filled image
    smoothed_image = gaussian_filter(image_filled, sigma=sigma)

    # 4. Create a binary mask where NaNs were, and apply the same filter
    mask = np.ones_like(image)
    mask[nan_mask] = 0
    smoothed_mask = gaussian_filter(mask, sigma=sigma)

    # 5. Combine the smoothed image with the smoothed mask
    result = smoothed_image / smoothed_mask

    # Optionally, restore NaNs to their original positions
    result[nan_mask] = np.nan

    return result


# Importing lookup table
n_bins = 25
with open(f"results/{run_mode}/correction_lookup_2d_{n_bins}_bins_lint.pkl", "rb") as f:
    correction_lookup_2d = pickle.load(f)
with open(f"results/{run_mode}/correction_lookup_3d_{n_bins}_bins_lint.pkl", "rb") as f:
    correction_lookup_3d = pickle.load(f)


# correction_lookup_3d["pe_mean"] consists of 25x25x25 images. I want them each smoothed using smooth_with_missing_values()


correction_lookup_3d_blurred = correction_lookup_3d.copy()
correction_lookup_3d_blurred["pe_mean"] = np.empty_like(correction_lookup_3d["pe_mean"])

correction_lookup_3d_blurred["pe_mean"] = smooth_with_missing_values(
    correction_lookup_3d["pe_mean"], 3
)

correction_lookup_3d_blurred["scaling"] = 1 / (
    1 + correction_lookup_3d_blurred["pe_mean"] / 100
)

correction_lookup_3d_blurred["scaling_lower"] = 1 / (
    1
    + (
        correction_lookup_3d_blurred["pe_mean"]
        + 1 * correction_lookup_3d_blurred["pe_std"]
    )
    / 100
)

correction_lookup_3d_blurred["scaling_upper"] = 1 / (
    1
    + (
        correction_lookup_3d_blurred["pe_mean"]
        - 1 * correction_lookup_3d_blurred["pe_std"]
    )
    / 100
)

# Error checking in case of std overpowering mean and leading to PEs < -100% and therefore negative scalings
# By replacing with pe_min, which is more negative for underestimation, we can ensure scaling regions are not negative
# and therefore do not invert the structure function

replacement_scaling = 1 / (1 + (correction_lookup_3d_blurred["pe_min"]) / 100)

correction_lookup_3d_blurred["scaling_upper"][
    correction_lookup_3d_blurred["scaling_upper"] < 0
] = replacement_scaling[correction_lookup_3d_blurred["scaling_upper"] < 0]


output_file_path = (
    f"results/{run_mode}/correction_lookup_{dim}d_{n_bins}_bins_lint_SMOOTHED.pkl"
)
with open(
    output_file_path,
    "wb",
) as f:
    pickle.dump(correction_lookup_3d_blurred, f)
print(f"Saved complete correction lookup table {output_file_path}")


# PLOT SMOOTHED 3D HEATMAPS

xedges = correction_lookup_3d["xedges"]
yedges = correction_lookup_3d["yedges"]
zedges = correction_lookup_3d["zedges"]
pe_mean = correction_lookup_3d_blurred["pe_mean"]


# Define the number of columns (you can adjust this as desired)
n_cols = 5  # Number of columns per row
n_rows = (n_bins + n_cols - 1) // n_cols  # Calculate number of rows needed

# MISSING VS LAG, BY POWER BIN
fig, ax = plt.subplots(
    n_rows,
    n_cols,
    figsize=(n_cols * 1.5, n_rows * 1.7),
    sharex=True,
    sharey=True,
)
plt.subplots_adjust(wspace=0.18, hspace=0.5)
plt.grid(False)
plt.suptitle(
    r"3D error heatmap: trend with increasing $\mathbf{power}$",
    y=0.98,
)

# Flatten the axis array to simplify indexing
ax = ax.flatten()

for i in range(n_bins):
    ax[i].grid(False)
    c = ax[i].pcolormesh(
        xedges,
        yedges,
        pe_mean[:, :, i],
        cmap="bwr",
    )
    c.set_clim(-100, 100)
    ax[i].set_title(
        f"({np.round(zedges[i], 2)},{np.round(zedges[i+1], 2)})",
    )
    ax[i].set_facecolor("black")
    ax[i].semilogx()

fig.text(
    0.5,
    0.03,
    "Lag ($\\tau$)",
    ha="center",
    va="center",
)  # Shared x-axis label
fig.text(
    0.05,
    0.5,
    "% missing",
    ha="center",
    va="center",
    rotation="vertical",
)  # Shared y-axis label

# Hide any extra subplots if n_bins is not a multiple of n_cols
for j in range(n_bins, len(ax)):
    fig.delaxes(ax[j])

# Add a color bar on the right-hand side of the figure, stretching down the entire height
cbar_ax = fig.add_axes(
    [0.92, 0.105, 0.02, 0.78]
)  # [left, bottom, width, height] to cover full height
cb = plt.colorbar(c, cax=cbar_ax)  # Attach the color bar to the last heatmap
cb.set_label("MPE")  # Optional: Label the color bar

plt.savefig(
    f"results/{run_mode}/plots/train_heatmap_{n_bins}bins_3d_lint_power_SMOOTHED.pdf",
    bbox_inches="tight",
)
plt.close()

# POWER VS % MISSING, BY LAG BIN
fig, ax = plt.subplots(
    n_rows,
    n_cols,
    figsize=(n_cols * 1.5, n_rows * 1.7),
    sharex=True,
    sharey=True,
)
plt.subplots_adjust(wspace=0.18, hspace=0.5)
plt.grid(False)
plt.suptitle(
    r"3D error heatmap: trend with increasing $\mathbf{lag}$",
    y=0.98,  # Was 1.02 for 2 rows
)

# Flatten the axis array to simplify indexing
ax = ax.flatten()

# Format lag bin edges to integers
formatted_xedges = [f"{x:.0f}".rstrip("0").rstrip(".") for x in xedges]

for i in range(n_bins):
    ax[i].grid(False)
    c = ax[i].pcolormesh(
        yedges,
        zedges,
        pe_mean[i, :, :],
        cmap="bwr",
    )
    c.set_clim(-100, 100)
    ax[i].set_title(
        f"({formatted_xedges[i]},{formatted_xedges[i+1]})",
    )
    ax[i].set_facecolor("black")
    ax[i].semilogy()

fig.text(
    0.5, 0.03, "% missing", ha="center", va="center"
)  # Shared x-axis label, was 0.00 y-val for 2 rows
fig.text(
    0.05,
    0.5,
    "Power",
    ha="center",
    va="center",
    rotation="vertical",
)  # Shared y-axis label

# Hide any extra subplots if n_bins is not a multiple of n_cols
for j in range(n_bins, len(ax)):
    fig.delaxes(ax[j])

# Add a color bar on the right-hand side of the figure, stretching down the entire height
cbar_ax = fig.add_axes(
    [0.92, 0.105, 0.02, 0.78]
)  # [left, bottom, width, height] to cover full height
cb = plt.colorbar(c, cax=cbar_ax)  # Attach the color bar to the last heatmap
cb.set_label("MPE")  # Optional: Label the color bar

plt.savefig(
    f"results/{run_mode}/plots/train_heatmap_{n_bins}bins_3d_lint_lag_SMOOTHED.pdf",
    bbox_inches="tight",
)
plt.close()

# POWER VS LAG, BIN % MISSING BIN
fig, ax = plt.subplots(
    n_rows,
    n_cols,
    figsize=(n_cols * 1.5, n_rows * 1.7),
    sharex=True,
    sharey=True,
)
plt.subplots_adjust(wspace=0.18, hspace=0.5)
plt.grid(False)
plt.suptitle(
    r"3D error heatmap: trend with increasing % $\mathbf{missing}$",
    y=0.98,
)
# Flatten the axis array to simplify indexing
ax = ax.flatten()
for i in range(n_bins):
    ax[i].grid(False)
    c = ax[i].pcolormesh(
        xedges,
        zedges,
        pe_mean[:, i, :],
        cmap="bwr",
    )
    c.set_clim(-100, 100)
    ax[i].set_title(
        f"({np.round(yedges[i], 2)},{np.round(yedges[i+1], 2)})",
    )
    ax[i].set_facecolor("black")
    ax[i].semilogx()
    ax[i].semilogy()
fig.text(0.5, 0.03, "Lag ($\\tau$)", ha="center", va="center")  # Shared x-axis label
fig.text(
    0.05,
    0.5,
    "Power",
    ha="center",
    va="center",
    rotation="vertical",
)  # Shared y-axis label

# Hide any extra subplots if n_bins is not a multiple of n_cols
for j in range(n_bins, len(ax)):
    fig.delaxes(ax[j])

# Add a color bar on the right-hand side of the figure, stretching down the entire height
cbar_ax = fig.add_axes(
    [0.92, 0.105, 0.02, 0.78]
)  # [left, bottom, width, height] to cover full height
cb = plt.colorbar(c, cax=cbar_ax)  # Attach the color bar to the last heatmap
cb.set_label("MPE")  # Optional: Label the color bar

plt.savefig(
    f"results/{run_mode}/plots/train_heatmap_{n_bins}bins_3d_lint_missing_SMOOTHED.pdf",
    bbox_inches="tight",
)
plt.close()

print("Heatmap smoothed and plots output")
