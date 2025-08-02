import matplotlib.pyplot as plt
import pandas as pd

import src.utils as utils  # copied directly from Reynolds project, normalize() added

# Set matplotlib font size
plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=6)

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

gap_handling_palette = {
    "true": "grey",
    "naive": "indianred",
    "lint": "#7570b3",
    "corrected_3d": "black",
}


# Run for with Wind and PSP

# df = pd.read_pickle("data/processed/psp/train/psp_fld_l2_mag_rtn_2018110200_v02.pkl")
df = pd.read_pickle("data/processed/wind/test/wi_h2_mfi_20160103_v05.pkl")

int_version = 1

# TIME SERIES
clean_subset = df["ints"][0][4000:4200]
fig, ax = plt.subplots(figsize=(1.5, 1))
ax.plot(clean_subset, lw=1)
ax.set_xticks([])
plt.show()

gapped_subset = df["ints_gapped"][
    (df["ints_gapped"].gap_handling == "naive")
    & (df["ints_gapped"].version == int_version)
][["Bx", "By", "Bz"]][3000:3100]

fig, ax = plt.subplots(figsize=(1.5, 1))
ax.plot(gapped_subset)
ax.set_xticks([])
plt.show()

lint_subset = df["ints_gapped"][
    (df["ints_gapped"].gap_handling == "lint")
    & (df["ints_gapped"].version == int_version)
][["Bx", "By", "Bz"]][3000:3100]

fig, ax = plt.subplots(figsize=(1.5, 1))
ax.plot(lint_subset, c="black", alpha=0.8, lw=0.8)
ax.plot(gapped_subset)
ax.set_xticks([])
plt.show()

# NORMALIZED TIME SERIES
clean_subset_normed = utils.normalize(clean_subset)
fig, ax = plt.subplots(figsize=(1.5, 1))
ax.plot(clean_subset_normed, lw=1)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)  # Mean line
ax.axhline(
    1, color="black", linestyle=":", linewidth=0.8, alpha=0.8
)  # Std deviation line
ax.axhline(
    -1, color="black", linestyle=":", linewidth=0.8, alpha=0.8
)  # Std deviation line
ax.set_xticks([])

plt.show()

# STRUCTURE FUNCTIONS
fig, ax = plt.subplots(figsize=(1.5, 1))
ax.loglog(df["sfs"]["sf_2"], lw=2, c="grey", alpha=0.7)
ax.set_xticks([])
ax.set_yticks([])
plt.show()

gapped_subset = df["sfs_gapped"][
    (df["sfs_gapped"].gap_handling == "naive")
    & (df["sfs_gapped"].version == int_version)
]

fig, ax = plt.subplots(figsize=(1.5, 1))
ax.loglog(gapped_subset["sf_2"], lw=1, c="indianred")
ax.set_xticks([])
ax.set_yticks([])
plt.show()

lint_subset = df["sfs_gapped"][
    (df["sfs_gapped"].gap_handling == "lint")
    & (df["sfs_gapped"].version == int_version)
]

fig, ax = plt.subplots(figsize=(1.5, 1))
ax.loglog(lint_subset["sf_2"], lw=1, c="#7570b3")
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# Just colouring true one black to pretend it's corrected

fig, ax = plt.subplots(figsize=(1.5, 1))
ax.loglog(df["sfs"]["sf_2"], lw=1, c="black")
ax.set_xticks([])
ax.set_yticks([])
plt.show()
