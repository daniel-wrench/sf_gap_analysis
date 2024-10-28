import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams.update(
    {
        "font.size": 10,  # Set font size to match LaTeX (e.g., 10pt)
        # "axes.labelsize": 10,  # Label size
        # "xtick.labelsize": 10,  # X-axis tick size
        # "ytick.labelsize": 10,  # Y-axis tick size
        # "legend.fontsize": 10,  # Legend font size
        # "figure.titlesize": 10,  # Figure title size
        # "figure.dpi": 300,  # Higher resolution figure output
    }
)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# Define min and max values for x, y, z
x_min, x_max = 0, 2000
y_min, y_max = 0, 2000
z_min, z_max = 0, 2000

# Generate 50 random 3D vectors within the specified ranges
num_vectors = 100
x_vals = np.random.uniform(x_min, x_max, num_vectors)
y_vals = np.random.uniform(y_min, y_max, num_vectors)
z_vals = np.random.uniform(z_min, z_max, num_vectors) / 100

# Stack into a single array of shape (50, 3)
vectors_3d = np.column_stack((x_vals, y_vals, z_vals))
vectors_3d

# Plotting
fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111, projection="3d")

# Plotting the scattered points
ax.scatter(
    vectors_3d[:, 0],
    vectors_3d[:, 1],
    vectors_3d[:, 2],
    color="blue",
    s=10,
    label="Random Points",
)


# Remove ticklabels
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Labels and legend
ax.set_xlabel("Lag ($\\tau$)", labelpad=50)
ax.set_ylabel("GP (\%)", labelpad=50)
ax.set_zlabel("SF ($\\tau$)", labelpad=50)

ax.xaxis.labelpad = -10
ax.yaxis.labelpad = -10
ax.zaxis.labelpad = -10

# Force there to be 25 ticks on the x-axis
ax.locator_params(axis="x", nbins=10)
ax.locator_params(axis="y", nbins=10)
ax.locator_params(axis="z", nbins=10)
plt.savefig("3d_scatter.pdf")
plt.show()
