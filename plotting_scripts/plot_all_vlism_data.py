# MAKE BACKGROUND PLOT: ALL VLISM DATA
# This script plots the VLISM and some heliosheath data from Voyager 1 and Voyager 2

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
# Set font size
plt.rcParams["font.size"] = 12


# Updated function: if doy_end and year_end are not provided,
# plot a single vertical line.
def plot_events(
    ax,
    name,
    doy_start,
    year_start,
    color="grey",
):
    # Compute the base date from the start parameters.
    date = pd.to_datetime(f"{year_start}-01-01") + pd.DateOffset(days=doy_start - 1)
    ylim = ax.get_ylim()
    # Plot a single vertical line at the specified date.
    ax.axvline(date, color=color, linestyle="--", alpha=0.7, label=name)
    # Place a label near the top of the line.
    ax.text(
        date + pd.DateOffset(days=5),
        ylim[1] * 0.92,
        name,
        rotation=90,
        verticalalignment="top",
        fontsize=9,
        color="black",
        alpha=0.7,
    )


# Add highlight regions for Voyager 1
v1_highlight_regions = [
    ("sh1", 335, 2012),
    ("sh2", 236, 2014),
    ("pf1", 346, 2016),
    ("pf2", 147, 2020),
]

v2_highlight_regions = [("pfa", 120, 2019), ("pfb", 244, 2019), ("sha", 180, 2020)]


v1_hp_date = pd.to_datetime("2012-08-25")
v2_hp_date = pd.to_datetime("2018-11-05")

# Read pickle files
v1_raw = pd.read_pickle("data/interim/voyager/voyager1_hs_lism.pkl")
v2_raw = pd.read_pickle("data/interim/voyager/voyager2_hs_lism.pkl")

v1_missing_fraction = v1_raw["BR"].isna().sum() / len(v1_raw)
v2_missing_fraction = v2_raw["BR"].isna().sum() / len(v2_raw)
print(f"Voyager 1 missing data fraction: {v1_missing_fraction:.2%}")
print(f"Voyager 2 missing data fraction: {v2_missing_fraction:.2%}")

df1 = v1_raw.resample("24h").mean()
df2 = v2_raw.resample("24h").mean()

# Compute the velocity of the v1 spacecraft, based on the Radius and datetime index
v1_duration = (df1.index[-1] - df1.index[0]).total_seconds()
v1_distance = df1["Radius"].iloc[-1] - df1["Radius"].iloc[0]
v1_velocity = v1_distance / v1_duration  # AU/s

v2_duration = (df2.index[-1] - df2.index[0]).total_seconds()
v2_distance = df2["Radius"].iloc[-1] - df2["Radius"].iloc[0]
v2_velocity = v2_distance / v2_duration  # AU/s


# Find common distance range for alignment
min_radius = df1.Radius.min()
max_radius = df1.Radius.max()

min_datetime_v1 = df1[df1.Radius > min_radius].index.min()
# Using the v2_velocity, compute the maximum datetime for v2
max_datetime_v1 = min_datetime_v1 + pd.Timedelta(
    (max_radius - min_radius) / v1_velocity, unit="s"
)

min_datetime_v2 = df2[df2.Radius > min_radius].index.min()
# Using the v2_velocity, compute the maximum datetime for v2
max_datetime_v2 = min_datetime_v2 + pd.Timedelta(
    (max_radius - min_radius) / v2_velocity, unit="s"
)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharey=True)

# Voyager 1 plot (top) - with paler pre-heliopause data

v1_metadata = {
    "F1": {"label": "$|\\bf{B}|$", "color": "black", "lw": 1.2, "sigma": 0.04},
    "BR": {"label": r"$B_R$", "color": "red", "lw": 0.5, "sigma": 0.06},
    # ^ this is the nominal value - could be between 0.02 and 0.1
    "BT": {"label": r"$B_T$", "color": "green", "lw": 0.5, "sigma": 0.02},
    "BN": {"label": r"$B_N$", "color": "blue", "lw": 0.5, "sigma": 0.02},
}

# Plot the Voyager 1 data with the specified metadata
for component, metadata in v1_metadata.items():

    # Pre-heliopause data (paler)
    pre_hp_mask = df1.index < v1_hp_date
    ax1.plot(
        df1.Radius[pre_hp_mask],
        df1[component][pre_hp_mask],
        color=metadata["color"],
        lw=metadata["lw"],
        alpha=0.4,
    )

    # Post-heliopause data (normal)
    post_hp_mask = df1.index >= v1_hp_date
    ax1.plot(
        df1.Radius[post_hp_mask],
        df1[component][post_hp_mask],
        color=metadata["color"],
        label=component,
        lw=metadata["lw"],
        alpha=1.0,
    )

    # Coordinates for the annotation
    # x_pos = df1.index[-1]
    # y_pos = df1[component][-1]
    # uncertainty_value = metadata["sigma"]

    # # Draw the error bar symbol (|-|) using a short vertical line with caps
    # ax1.errorbar(
    #     x_pos, y_pos, yerr=uncertainty_value, fmt="o", color="black", capsize=5
    # )

    # # Add text label with ± and value next to it
    # ax1.text(x_pos, y_pos, f"±{uncertainty_value}", fontsize=12, va="center")


ax1.set_ylabel("Magnetic Field Strength (nT)")

handles, labels = ax1.get_legend_handles_labels()
# handles = [handles[0], handles[2], handles[1]]
labels = ["$|\\bf{B}|$", r"$B_R$", r"$B_T$", r"$B_N$"]

ax2.legend(handles, labels, loc="center", fontsize=14)


# Add secondary x-axis for dates (Voyager 1)
ax1_date = ax1.twiny()
ax1_date.plot(df1.index, df1["BR"], alpha=0)
ax1_date.set_xlabel("Date")


v2_metadata = {
    "F1": {"label": "$|\\bf{B}|$", "color": "black", "lw": 1.2, "sigma": 0.04},
    "BR": {"label": r"$B_R$", "color": "red", "lw": 0.5, "sigma": 0.06},
    # ^ this is the nominal value - could be between 0.02 and 0.1
    "BT": {"label": r"$B_T$", "color": "green", "lw": 0.5, "sigma": 0.03},
    "BN": {"label": r"$B_N$", "color": "blue", "lw": 0.5, "sigma": 0.03},
}

# Voyager 2 plot (bottom) - with paler pre-heliopause data
for component, metadata in v2_metadata.items():
    # Pre-heliopause data (paler)
    pre_hp_mask = df2.index < v2_hp_date
    ax2.plot(
        df2.index[pre_hp_mask],
        df2[component][pre_hp_mask],
        label=metadata["label"],
        color=metadata["color"],
        lw=metadata["lw"],
        alpha=0.4,
    )

    # Post-heliopause data (normal)
    post_hp_mask = df2.index >= v2_hp_date
    ax2.plot(
        df2.index[post_hp_mask],
        df2[component][post_hp_mask],
        color=metadata["color"],
        lw=metadata["lw"],
        alpha=1.0,
    )


ax2.set_xlabel("Date")
ax2.set_ylabel("Magnetic Field Strength (nT)")


# Add secondary x-axis for dates (Voyager 2)
# Add secondary x-axis for distance (Voyager 2) - NOW SHOWS DISTANCE
ax2_date = ax2.twiny()
ax2_date.plot(df2.Radius, df2["BR"], alpha=0)  # Changed from df2.index

ax1_date.axvline(v1_hp_date, color="k", linestyle="--", lw=2)
ax2.axvline(v2_hp_date, color="k", linestyle="--", lw=2)

# Align primary x-axes (distance)
ax1.set_xlim(min_radius, max_radius)  # Extend a bit for better visibility
# ax2.set_xlim(min_radius, max_radius)

ax1_date.set_xlim(min_datetime_v1, max_datetime_v1)
# ax2_date.set_xlim(min_datetime_v2, max_datetime_v2)

# Swap the xlim assignments for ax2 and ax2_date
ax2.set_xlim(min_datetime_v2, max_datetime_v2)  # Now dates on primary
ax2_date.set_xlim(min_radius, max_radius)  # Now distance on secondary

ax1.set_ylim(-0.7, 1)

for region in v2_highlight_regions:
    plot_events(ax2, *region)

for region in v1_highlight_regions:
    plot_events(ax1_date, *region)


ax1_date.text(pd.to_datetime("2021-03-01"), 0.6, "hump", alpha=0.7, fontsize=9)

ax1_date.text(
    v1_hp_date - pd.DateOffset(days=200),
    ax1_date.get_ylim()[1] * 0.92,
    "HP",
    rotation=90,
    verticalalignment="top",
    fontsize=14,
    color="black",
    alpha=0.8,
    fontweight="bold",
)

ax2.text(
    v2_hp_date - pd.DateOffset(days=220),
    ax2.get_ylim()[1] * 0.92,
    "HP",
    rotation=90,
    verticalalignment="top",
    fontsize=14,
    color="black",
    alpha=0.8,
    fontweight="bold",
)

ax2_date.set_xticklabels([])

ax1.annotate(
    "Voyager 1",
    xy=(0.98, 0.85),
    xycoords="axes fraction",
    fontsize=16,
    fontweight="bold",
    ha="right",
    va="bottom",
)
ax2.annotate(
    "Voyager 2",
    xy=(0.98, 0.85),
    xycoords="axes fraction",
    fontsize=16,
    fontweight="bold",
    ha="right",
    va="bottom",
)

handles, labels = ax1.get_legend_handles_labels()
labels = ["$|\\bf{B}|$", r"$B_R$", r"$B_T$", r"$B_N$"]

# Add vertical gridlines for Voyager 1 and Voyager 2
ax1.grid(True, which="both", axis="x", alpha=0.6)
ax2_date.grid(True, which="both", axis="x", alpha=0.6, zorder=1)

ax1.tick_params(axis="x", which="major", pad=15)
for label in ax1.get_xticklabels():
    label.set_fontweight("bold")
# ax1.set_xlabel("DISTANCE FROM SUN (AU)")

# Add annotation inside a box
ax1.annotate(
    "Distance from Sun (au)",
    xy=(0.48, -0.18),
    xycoords="axes fraction",
    fontsize=12,
    fontweight="bold",
    ha="center",
    va="bottom",
    bbox=dict(facecolor="white", alpha=1, edgecolor="white"),
)

ax2.legend(
    handles,
    labels,
    loc="center right",
    fontsize=13,
    ncol=2,
    frameon=True,
    facecolor="white",  # legend box background color
    edgecolor="black",  # legend box border color
    framealpha=1.0,  # legend box transparency
)


# Reduce spacing between subplots
plt.subplots_adjust(hspace=0.25)  # Removed to avoid conflict with tight_layout

# plt.tight_layout()
plt.savefig("bg_figs/bg_all_vlism_data.png", dpi=300, bbox_inches="tight")


# Drop a column from df1
df1.drop(columns=["Radius"], inplace=True)


# Plot Voyager 2 post-heliopause data
ax_v2 = df2.loc[df2.index > v2_hp_date].plot(lw=0.5, figsize=(8, 3))
plt.axvline("2021-01-01", color="k", linestyle="--", lw=1, alpha=0.5)
plt.title("Voyager 2 Post-Heliopause 48s MAG Data")

# Re-order the legend
handles, labels = ax_v2.get_legend_handles_labels()
order = ["F1", "BN", "BR", "BT"]
ordered_handles = [handles[labels.index(k)] for k in order if k in labels]
ordered_labels = order
ax_v2.legend(ordered_handles, ordered_labels, loc="best")
