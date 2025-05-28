import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from scipy.spatial.distance import pdist, squareform

# Set the random seed for reproducibility
np.random.seed(42)

# Generate synthetic time series data with uncertainty
n_points = 50
t = np.linspace(0, 10, n_points)
# True signal: a sine wave with some trend
true_signal = 2 * np.sin(t) + 0.2 * t
# Adding heteroscedastic measurement errors (arrray of constant error for simplicity)
error_magnitude = np.random.uniform(0.5, 1.5, size=n_points)
measured_values = true_signal + np.random.normal(0, error_magnitude)


# Calculate variogram cloud with uncertainty propagation and keep track of point pairs
def calculate_variogram_cloud_with_uncertainty(times, values, errors):
    n = len(times)
    # All pairwise distances between time points
    h_pairs = squareform(pdist(times.reshape(-1, 1)))
    # Flatten to get distances for the variogram cloud
    h = h_pairs.flatten()

    # Calculate squared differences for each pair of points
    gamma = np.zeros(n * n)
    gamma_err = np.zeros(n * n)
    # Keep track of indices for each pair
    pair_indices = np.zeros((n * n, 2), dtype=int)

    k = 0
    for i in range(n):
        for j in range(n):
            # Squared difference between values
            gamma[k] = 0.5 * (values[i] - values[j]) ** 2

            # Error propagation for the squared difference
            # Using error propagation formula for (x-y)²:
            # Var[(x-y)²] ≈ (x-y)²*(σx²+σy²)
            gamma_err[k] = np.sqrt(
                (values[i] - values[j]) ** 2 * (errors[i] ** 2 + errors[j] ** 2)
            )

            # Store the indices of the pair
            pair_indices[k, 0] = i
            pair_indices[k, 1] = j

            k += 1

    # Remove self-comparisons (where h=0)
    mask = h > 0
    h = h[mask]
    gamma = gamma[mask]
    gamma_err = gamma_err[mask]
    pair_indices = pair_indices[mask]

    return h, gamma, gamma_err, pair_indices


# Calculate variogram cloud
h, gamma, gamma_err, pair_indices = calculate_variogram_cloud_with_uncertainty(
    t, measured_values, error_magnitude
)


# Calculate binned variogram (for the smooth line)
def bin_variogram(h, gamma, bin_edges):
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_means = np.zeros(len(bin_centers))

    for i in range(len(bin_centers)):
        mask = (h >= bin_edges[i]) & (h < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_means[i] = np.mean(gamma[mask])
        else:
            bin_means[i] = np.nan

    return bin_centers, bin_means


# Create bin edges for variogram averaging
bin_edges = np.linspace(0, max(h), 15)
bin_centers, bin_means = bin_variogram(h, gamma, bin_edges)

# Select a few representative points to highlight (based on distance ranges)
h_ranges = [
    (0, 2),  # Very short distance
    (2, 4),  # Short distance
    (4, 6),  # Medium distance
    (6, 8),  # Long distance
    (8, 10),  # Very long distance
]

# Find one point from each range
highlighted_indices = []
for h_min, h_max in h_ranges:
    mask = (h >= h_min) & (h < h_max)
    if np.sum(mask) > 0:
        # Get the index of a point in this range (pick the middle one for representativeness)
        indices = np.where(mask)[0]
        highlighted_indices.append(indices[len(indices) // 2])

# Define colors for each highlighted point
highlight_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

# Create the plot with highlighted connections
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(12, 12))
gs = GridSpec(2, 1, height_ratios=[1, 1.2])

# Time series plot with error bars
ax1 = fig.add_subplot(gs[0])
ax1.errorbar(
    t,
    measured_values,
    yerr=error_magnitude,
    fmt="o",
    color="#1f77b4",
    alpha=0.7,
    ecolor="gray",
    capsize=3,
    label="Measured Values",
)
ax1.plot(t, true_signal, "k-", alpha=0.8, label="True Signal")
ax1.set_xlabel("Time", fontsize=12)
ax1.set_ylabel("Value", fontsize=12)
ax1.set_title("Time Series with Measurement Uncertainty", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Variogram cloud plot with error bars
ax2 = fig.add_subplot(gs[1])

# Create colormap based on distance
norm = mpl.colors.Normalize(vmin=min(h), vmax=max(h))
cmap = plt.cm.viridis

# Plot all variogram cloud points with error bars (low alpha)
for i in range(len(h)):
    ax2.errorbar(
        h[i],
        gamma[i],
        yerr=gamma_err[i],
        fmt="o",
        color=cmap(norm(h[i])),
        alpha=0.2,
        ecolor="gray",
        capsize=2,
        markersize=4,
    )

# Plot the smooth variogram line
ax2.plot(bin_centers, bin_means, "k-", linewidth=2, label="Experimental Variogram")

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2)
cbar.set_label("Distance (h)", fontsize=12)

ax2.set_xlabel("Distance (h)", fontsize=12)
ax2.set_ylabel("Semivariance γ(h)", fontsize=12)
ax2.set_title("Variogram Cloud with Propagated Uncertainty", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Highlight selected points and draw connections
for idx, color in zip(highlighted_indices, highlight_colors):
    # Get indices of the time series points that created this variogram point
    i, j = pair_indices[idx]

    # Highlight in the variogram
    ax2.errorbar(
        h[idx],
        gamma[idx],
        yerr=gamma_err[idx],
        fmt="o",
        color=color,
        alpha=1.0,
        ecolor=color,
        capsize=4,
        markersize=8,
        zorder=10,
        label=f"Pair {i+1}-{j+1}, d={h[idx]:.2f}",
    )

    # Highlight in the time series
    ax1.plot(t[i], measured_values[i], "o", color=color, markersize=10, zorder=10)
    ax1.plot(t[j], measured_values[j], "o", color=color, markersize=10, zorder=10)

    # Draw connecting line in time series
    ax1.plot(
        [t[i], t[j]],
        [measured_values[i], measured_values[j]],
        "--",
        color=color,
        linewidth=2,
        alpha=0.7,
        zorder=5,
    )

    # Add annotations for the points
    ax1.annotate(
        f"{i+1}",
        (t[i], measured_values[i]),
        xytext=(5, 5),
        textcoords="offset points",
        color=color,
        fontweight="bold",
    )
    ax1.annotate(
        f"{j+1}",
        (t[j], measured_values[j]),
        xytext=(5, 5),
        textcoords="offset points",
        color=color,
        fontweight="bold",
    )

    # Draw curves connecting the time series to variogram
    # Calculate bezier curve control points
    vg_x, vg_y = h[idx], gamma[idx]  # Variogram point
    ts_x = (t[i] + t[j]) / 2  # Midpoint of time series x
    ts_y = measured_values[i]  # Use first point's y for visualization clarity

    # Create curved connecting line using bezier path
    y_mid = (ax1.get_position().y0 + ax2.get_position().y1) / 2

    # Convert data coords to figure coords for drawing between subplots
    ts_disp = ax1.transData.transform((ts_x, ts_y))
    vg_disp = ax2.transData.transform((vg_x, vg_y))
    ts_fig = fig.transFigure.inverted().transform(ts_disp)
    vg_fig = fig.transFigure.inverted().transform(vg_disp)

    # Create control points for the curve
    control_x = (ts_fig[0] + vg_fig[0]) / 2

    # Draw the connecting curve in figure coordinates
    curve = Path(
        [
            (ts_fig[0], ts_fig[1]),  # Start
            (control_x, y_mid),  # Control point
            (vg_fig[0], vg_fig[1]),  # End
        ],
        [Path.MOVETO, Path.CURVE3, Path.CURVE3],
    )

    patch = patches.PathPatch(
        curve,
        facecolor="none",
        edgecolor=color,
        linewidth=1.5,
        alpha=0.7,
        linestyle=":",
        transform=fig.transFigure,
    )
    fig.patches.append(patch)

# Add legend to the variogram plot for the highlighted points
ax2.legend(fontsize=9, loc="upper left")

# Add an explanatory text box
textstr = "Colored points show the connection between:\n"
textstr += "• Pairs of time series points (top)\n"
textstr += "• Their corresponding variogram points (bottom)\n"
textstr += "Points with the same color are linked\n"
textstr += "Dotted lines show the connections"

props = dict(boxstyle="round", facecolor="wheat", alpha=0.7)
ax1.text(
    0.02,
    0.02,
    textstr,
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="bottom",
    bbox=props,
)

plt.tight_layout()
plt.savefig("variogram_connections_static.png", dpi=300, bbox_inches="tight")
plt.show()
