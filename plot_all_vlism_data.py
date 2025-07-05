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


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

# Voyager 1 plot (top)
for component, color, lw in zip(
    ["F1", "BR", "BT", "BN"], ["black", "red", "green", "blue"], [0.9, 0.2, 0.2, 0.2]
):
    ax1.plot(df1.Radius, df1[component], label=component, color=color, lw=lw)

ax1.set_xlabel("Distance from the Sun (AU)")
ax1.set_ylabel("Magnetic Field Strength (nT)")

handles, labels = ax1.get_legend_handles_labels()
# handles = [handles[0], handles[2], handles[1]]
labels = ["$|\\bf{B}|$", r"$B_R$", r"$B_T$", r"$B_N$"]

ax1.legend(handles, labels, loc="upper left", fontsize=10)

ax1.annotate(
    "VOYAGER 1",
    xy=(0.99, 0.9),
    xycoords="axes fraction",
    fontsize=16,
    fontweight="bold",
    ha="right",
    va="bottom",
)

# Add secondary x-axis for dates (Voyager 1)
ax1_date = ax1.twiny()
ax1_date.plot(df1.index, df1["BR"], alpha=0)
ax1_date.set_xlabel("Date")

# Voyager 2 plot (bottom)
for component, color, lw in zip(
    ["F1", "BR", "BT", "BN"], ["black", "red", "green", "blue"], [0.9, 0.2, 0.2, 0.2]
):
    ax2.plot(df2.Radius, df2[component], label=component, color=color, lw=lw)

ax2.set_xlabel("Distance from the Sun (AU)")
ax2.set_ylabel("Magnetic Field Strength (nT)")

# Add secondary x-axis for dates (Voyager 2)
ax2_date = ax2.twiny()
ax2_date.plot(df2.index, df2["BR"], alpha=0)
ax2_date.set_xlabel("Date")

ax1_date.axvline(v1_hp_date, color="k", linestyle="--")
ax2_date.axvline(v2_hp_date, color="k", linestyle="--")


plt.tight_layout()
plt.savefig("bg_figs/bg_all_vlism_data.png", dpi=300, bbox_inches="tight")


df2.head()
