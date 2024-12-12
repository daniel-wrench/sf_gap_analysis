import pickle

import matplotlib.pyplot as plt
import numpy as np

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

# Parameters
fig = plt.figure(figsize=(3.5, 3.5))  # Increased figure size for better visualization
dim = 3
n_bins = 25
output_path = "final"  # Ensure to set this properly

# Load correction lookup data
with open(
    f"data/corrections/{output_path}/correction_lookup_{dim}d_{n_bins}_bins.pkl",
    "rb",
) as f:
    correction_lookup = pickle.load(f)

xedges = correction_lookup["xedges"]
yedges = correction_lookup["yedges"]
zedges = correction_lookup["zedges"]
pe_mean = correction_lookup["pe_mean"]

# Define positions and data for each overlapping panel
panels = [
    [0.4, 0.3, 0.3, 0.3, 21],  # [x, y, width, height, z-index]
    [0.3, 0.2, 0.3, 0.3, 15],
    [0.2, 0.1, 0.3, 0.3, 5],
]

# Draw overlapping panels with plots
for x, y, w, h, i in panels:
    ax = fig.add_axes([x, y, w, h])  # Add subplot at specific position

    # Create a heatmap
    c = ax.pcolormesh(xedges[2:], yedges[2:], pe_mean[2:, 2:, i], cmap="bwr")
    c.set_clim(-100, 100)  # Set color limits

    # Annotate the z-range
    ax.annotate(
        f"[{np.round(zedges[i], 2)}, {np.round(zedges[i+1], 2)}]",
        xy=(0.5, 0.85),
        xycoords="axes fraction",
        color="black",
        ha="center",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            # reduce padding and remove border
            pad=0.5,
            edgecolor="none",
        ),
    )

    # Set axis labels for the final panel (bottom-left)
    if i == 5:
        ax.set_ylabel("GP (\%)")
        ax.set_xlabel("Lag ($\\tau$)")
        ax.annotate(
            r"$\hat{S}_2^{\mathrm{LINT}}$ bin range",
            xy=(0.5, 0.7),
            xycoords="axes fraction",
            color="white",
            fontsize=6,
            ha="center",
        )

    # Set log-scale for x-axis
    ax.semilogx()

    # Style adjustments
    ax.set_facecolor("black")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
        spine.set_linewidth(1)

    ax.tick_params(axis="both", color="gray")  # Applies to both x and y axes
    # Remove minor ticks
    ax.minorticks_off()

# Add text to the figure, with a 45 degree rotation
# fig.text(
#     0.7,
#     0.15,
#     r"$\hat{S}_2^{\mathrm{LINT}}$",
#     ha="center",
#     va="center",
#     rotation=35,
#     color="black",
# )

cbar_ax = fig.add_axes(
    [0.72, 0.08, 0.02, 0.52]
)  # [left, bottom, width, height] to cover full height
cb = plt.colorbar(c, cax=cbar_ax)  # Attach the color bar to the last heatmap
cb.set_label("MPE (\%)")  # Optional: Label the color bar

# Save the plot
plt.savefig(f"plots/results/{output_path}/3d_heatmap.png", bbox_inches="tight")
