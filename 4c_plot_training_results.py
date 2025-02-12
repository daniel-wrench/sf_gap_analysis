import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.params as params

plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams.update(
    {
        "font.size": 10,  # Set font size to match LaTeX (e.g., 10pt)
        "axes.labelsize": 10,  # Label size
        "xtick.labelsize": 10,  # X-axis tick size
        "ytick.labelsize": 10,  # Y-axis tick size
        "legend.fontsize": 10,  # Legend font size
        "figure.titlesize": 10,  # Figure title size
        "figure.dpi": 300,  # Higher resolution figure output
    }
)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# Suppress the specific RankWarning from numpy - occurs with fitting slope sometimes
warnings.filterwarnings("ignore", category=np.RankWarning)

np.random.seed(123)  # For reproducibility

run_mode = params.run_mode
times_to_gap = params.times_to_gap

spacecraft = "psp"

# Import all corrected (test) files
n_bins = 25
# times_to_gap = params.times_to_gap # removing as will only be using this file locally

data_path_prefix = params.data_path_prefix

# Also do publication-ready plots for heatmaps
dim = 2

# Read in binned errors for FULL SET of training intervals
with open(
    f"results/{run_mode}/correction_lookup_{dim}d_{n_bins}_bins_naive.pkl",
    "rb",
) as f:
    correction_lookup_naive = pickle.load(f)

with open(
    f"results/{run_mode}/correction_lookup_{dim}d_{n_bins}_bins_lint.pkl",
    "rb",
) as f:
    correction_lookup_lint = pickle.load(f)

# Read in lag-specific errors for SUBSET of training intervals
sfs_gapped = pd.read_pickle("data/processed/train_psp_sfs_gapped.pkl")

# Print the number of unique int_index-file_index combinations
print(
    f"Number of unique int_index-file_index combinations (raw intervals): {sfs_gapped[['int_index', 'file_index']].drop_duplicates().shape[0]}"
)
print(
    f"Total number of intervals: {sfs_gapped[['int_index', 'file_index', 'version']].drop_duplicates().shape[0]}"
)


df = sfs_gapped
estimator = "sf_2"

xedges = correction_lookup_lint["xedges"]
yedges = correction_lookup_lint["yedges"]
pe_mean_lint = correction_lookup_lint["pe_mean"]
pe_mean_naive = correction_lookup_naive["pe_mean"]

###################################


# Create a 2x2 figure layout
fig, ax = plt.subplots(
    figsize=(5.5, 5), nrows=2, ncols=2, sharey="row", sharex="col", tight_layout=True
)
plt.grid(True)

# --- First Row (Scatter plots) ---

# First scatter plot (Naive)
other_outputs_df_naive = df[df["gap_handling"] == "naive"]
ax[0, 0].scatter(
    other_outputs_df_naive["lag"],
    other_outputs_df_naive[estimator + "_pe"],
    c=other_outputs_df_naive["missing_percent"],
    s=0.03,
    alpha=0.4,
    cmap="plasma",
)

mean_error_naive = other_outputs_df_naive.groupby("lag")[estimator + "_pe"].mean()
ax[0, 0].plot(mean_error_naive, color="black", lw=3, label="Ensemble average")
ax[0, 0].set_title("Naive", fontsize=10)
ax[0, 1].set_title("LINT", fontsize=10)

ax[0, 0].hlines(
    0,
    1,
    other_outputs_df_naive.lag.max(),
    color="grey",
    linestyle="--",
    lw=1.5,
    label="0\% error",
)
ax[0, 0].set_ylabel("PE (\%)")
ax[0, 0].semilogx()
ax[0, 0].set_xlim(1, params.max_lag_prop * params.int_length)

# Annotate the MAPE for naive
ax[0, 0].annotate(
    "MAPE = {0:.2f}".format(other_outputs_df_naive[estimator + "_pe"].abs().mean()),
    xy=(1, 1),
    xycoords="axes fraction",
    xytext=(0.1, 0.85),
    textcoords="axes fraction",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="grey", facecolor="white", alpha=0.5),
)

# Annotate the MAPE for LINT
# ax[0, 0].annotate(
#     "MPE = {0:.2f}".format(other_outputs_df_naive[estimator + "_pe"].mean()),
#     xy=(1, 1),
#     xycoords="axes fraction",
#     xytext=(0.1, 0.9),
#     textcoords="axes fraction",
#     c="black",
# )

legend = ax[0, 0].legend(loc="upper left")
legend.get_frame().set_alpha(0.5)  # Set the box to be semi-transparent
ax[0, 0].set_ylim(-100, 100)

# Second scatter plot (LINT)
other_outputs_df_lint = df[df["gap_handling"] == "lint"]

sc3 = ax[0, 1].scatter(
    other_outputs_df_lint["lag"],
    other_outputs_df_lint[estimator + "_pe"],
    c=other_outputs_df_lint["missing_percent"],
    s=0.03,
    alpha=0.4,
    cmap="plasma",
)
mean_error_lint = other_outputs_df_lint.groupby("lag")[estimator + "_pe"].mean()
ax[0, 1].plot(mean_error_lint, color="black", lw=3, label="MPE($\\tau$)")

# Annotate the MAPE for LINT
ax[0, 1].annotate(
    "MAPE = {0:.2f}".format(other_outputs_df_lint[estimator + "_pe"].abs().mean()),
    xy=(1, 1),
    xycoords="axes fraction",
    xytext=(0.1, 0.85),
    textcoords="axes fraction",
    c="black",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="grey", facecolor="white", alpha=0.5),
)

ax[0, 1].hlines(
    0,
    1,
    other_outputs_df_lint.lag.max(),
    color="grey",
    linestyle="--",
    label="0\% error",
)
ax[0, 1].set_ylim(-100, 100)
ax[0, 1].semilogx()
ax[0, 0].legend(loc="lower left")
ax[0, 1].set_xlim(1, params.max_lag_prop * params.int_length)

# Colorbar for the second row
cb2 = plt.colorbar(sc3, cax=ax[0, 1].inset_axes([1.05, 0, 0.03, 1]))
sc3.set_clim(0, 100)
cb2.set_label("TGP \%")

# --- Second Row (Heatmaps) ---

# Plot the heatmap for the first column
sc = ax[1, 0].pcolormesh(
    xedges,
    yedges,
    pe_mean_naive.T,
    cmap="bwr",
)
ax[1, 0].grid(False)
ax[1, 0].set_xlabel("Lag ($\\tau$)")
ax[1, 0].set_ylabel("GP (\%)")
ax[1, 0].set_facecolor("black")
ax[1, 0].set_xscale("log")
ax[1, 0].set_xlim(1, params.max_lag_prop * params.int_length)


# Plot the heatmap for the second column
sc2 = ax[1, 1].pcolormesh(
    xedges,
    yedges,
    pe_mean_lint.T,
    cmap="bwr",
)
ax[1, 1].grid(False)
ax[1, 1].set_xlabel("Lag ($\\tau$)")
ax[1, 1].set_facecolor("black")
ax[1, 1].set_xscale("log")
ax[1, 1].set_xlim(1, params.max_lag_prop * params.int_length)

# Colorbar for the first row
cb1 = plt.colorbar(sc, cax=ax[1, 1].inset_axes([1.05, 0, 0.03, 1]))
sc.set_clim(-100, 100)
sc2.set_clim(-100, 100)
cb1.set_label("MPE \%")

fig.align_ylabels(ax[:, 0])
# Align the colorbar labels (currently slightly offset)
cb1.ax.yaxis.set_label_coords(6, 0.5)
cb2.ax.yaxis.set_label_coords(6, 0.5)


plt.subplots_adjust(wspace=0.05, hspace=0.1)
# plt.show()
# GIANT - DO NOT SAVE AS PDF
plt.savefig(
    f"results/{run_mode}/plots/train_psp_error.png",
)

# sys.exit()


# NOW PLOT 3D HEATMAPS - NOT CURRENTLY USED!

dim = 3
n_bins = 25

with open(
    f"results/{run_mode}/correction_lookup_{dim}d_{n_bins}_bins_lint.pkl",
    "rb",
) as f:
    correction_lookup = pickle.load(f)

xedges = correction_lookup["xedges"]
yedges = correction_lookup["yedges"]
zedges = correction_lookup["zedges"]
pe_mean = correction_lookup["pe_mean"]

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
    "GP (\%)",
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
cb.set_label("MPE (\%)")  # Optional: Label the color bar
# plt.savefig(
#     f"plots/results/{run_mode}/train_heatmap_{n_bins}bins_3d_lint_power.pdf",
#     bbox_inches="tight",
# )
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
    0.5, 0.03, "GP (\%)", ha="center", va="center"
)  # Shared x-axis label, was 0.00 y-val for 2 rows
fig.text(
    0.05,
    0.5,
    r"\hat{S}_2^\text{LINT}(\tau)",
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
# plt.show()
# plt.savefig(
#     f"plots/results/{run_mode}/train_heatmap_{n_bins}bins_3d_lint_lag.pdf",
#     bbox_inches="tight",
# )
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
    r"3D error heatmap: trend with increasing \% $\mathbf{missing}$",
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
    r"\hat{S}_2^\text{LINT}(\tau)",
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
# plt.show()
# plt.savefig(
#     f"plots/results/{run_mode}/train_heatmap_{n_bins}bins_3d_lint_missing.pdf",
#     bbox_inches="tight",
# )
plt.close()
