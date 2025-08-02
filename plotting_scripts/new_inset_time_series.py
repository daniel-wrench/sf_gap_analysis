import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in"})
# Adjust font size
plt.rcParams.update({"font.size": 12})

# Read in cleaned Voyager 1 data
df = pd.read_pickle("data/interim/voyager/voyager1_lism_cleaned.pkl")
print("Loaded dataset")

import matplotlib.dates as mdates

df["F1"]["2018-01-01":"2018-01-07"].plot(lw=0.3, figsize=(8, 2), ylim=(0.38, 0.48))
ax.set_ylabel("$B$ (nT)")

# Format x-axis for dates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%d"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator())
# fig.autofmt_xdate()

plt.tight_layout()

data = df["F1"]["2018":"2019"]

# zoom_values: [min_value, max_value]

fig, ax = plt.subplots(figsize=(8, 6))


# Main plot
ax.plot(data, linewidth=1)
ax.set_ylabel("$B$ (nT)")
ax.grid(True, alpha=0.3)

# Create inset
axins = ax.inset_axes([0.6, 0.8, 0.35, 0.2])  # [x, y, width, height]

# Plot zoomed data
# zoom_mask = (time_data >= zoom_region[0]) & (time_data <= zoom_region[1])
masked_data = data["2018-01-01":"2018-01-04"]
axins.plot(masked_data, color="k", linewidth=0.3)
# axins.plot(time_data[zoom_mask], signal_data[zoom_mask], "k-", linewidth=0.3)
# axins.set_xlim(zoom_region)
# axins.set_ylim(zoom_values)
axins.grid(True, alpha=0.3)
axins.tick_params(labelsize=8)

# Indicate the region of the inset
# ax.indicate_inset_zoom(axins, edgecolor="red", linewidth=1.5)
