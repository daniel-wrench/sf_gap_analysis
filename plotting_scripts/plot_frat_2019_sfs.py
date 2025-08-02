# # VOYAGER DATA


# This script has two purposes:
#
# 1. **Reproduce a particularly messy structure function in FraternaleEA (2019)**, including his statistical threshold. *I have pretty much reproduced the structure functions from the inner heliosheath (V1, 2011), though at a reduced frequency due to computation time.*
# 2. **Apply my correction factor to an interval from Voyager.** A LISM interval would be coolest, but that paper does not have SFs of that region to compare with, only spectra (for which they use their suite of spectral estimation techniques). That said, I will also be converting the SF to an E.S. In either case, there may be issues around interval length (so as to remain consistent with my standardised intervals) that make direct comparisons difficult. (See comments in markdown section "Issues with comparing results".) *I have so far done this for both (need to reproduce both, but need to finish 3D smoothing)*


import math as m
import os
import sys

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt
from sunpy.timeseries import TimeSeries

# Import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.params as params
import src.sf_funcs as sf
import src.utils as utils

# Set matplotlib font size
plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


def SmoothySpec(a, nums=None):
    """Smooth a curve using a moving average smoothing"""
    b = a.copy()
    if nums is None:
        nums = 2 * len(b) // 3
    for i in range(nums):
        b[i + 1 : -1] = 0.25 * b[i:-2] + 0.5 * b[i + 1 : -1] + 0.25 * b[i + 2 :]
    return b


# # Fraternale et al. (2019)


spacecraft = "voyager"
cadence = "48s"


print(f"Reading in {spacecraft} data...")
data = TimeSeries(
    "data/raw/voyager/voyager1_48s_mag-vim_20110101_v01.cdf",
    concatenate=True,  # Not strictly relevant here as only reading one file at a time
)

df_raw = data.to_dataframe()

df_raw = df_raw.loc[:, params.mag_vars_dict[spacecraft]]

df = df_raw.resample(cadence).mean()


# Proportion of missing data
print(df.isna().sum() / len(df["F1"]))

# Get DOY 276-365 from the raw data
# subset = df["2021-02-01":"2021-05-01"]
subset = df["2011-10-03":"2011-12-31"]

del df

subset.isna().sum() / len(subset)
# 80% data missing


# ### Computing full SFs


subset_resampled = subset.resample("10min").mean()  # Now only 12,000 long
len(subset_resampled)


# Takes about 20-30 seconds,

sf_full = utils.calc_struct_sdk(
    data=subset_resampled["BT"],
    freq=1 / 600,
    orders=[1, 2, 3, 4],
    max_lag_fraction=0.5,
    plot=False,
)


subset_lint = subset_resampled.interpolate(method="linear")

sf_full_lint = utils.calc_struct_sdk(
    data=subset_lint["BT"],
    freq=1 / 600,
    orders=[1, 2, 3, 4],
    max_lag_fraction=0.5,
    plot=False,
)


# ### Thresholding procedure
# "the computation of $S$ is nontrivial for Voyager data sets due to the amount and distribution of missing data."
# They do not interpolate, but they do account for the variable reduction in sample size at each lag by applying a
# threshold of statistical significance as to whether they use certain lags to calculate the slopes
#


# Define the window size for 48 hours in seconds
# Convert index to time index
sf_full.index = pd.to_timedelta(sf_full.index, unit="s")

window_size = pd.to_timedelta("48h")

# Compute the rolling maximum for a 48-hour window
sf_full["rolling_max"] = (
    sf_full["N"].rolling(window=window_size, min_periods=1, center=True).max()
)

# Compute the threshold
sf_full["threshold"] = 0.25 * sf_full["rolling_max"]

# Determine the color based on the threshold
sf_full["color"] = np.where(sf_full["N"] < sf_full["threshold"], "gray", "black")

# Convert the time index to back to seconds
sf_full.index = sf_full.index.total_seconds()


# ### Computing correlation scale


# time_lags_lr, r_vec_lr = utils.compute_nd_acf(
#     [subset.BR, subset.BT, subset.BN],
#     nlags=10000,  # Arbritrary large number
#     plot=False,
# )
# tc_exp = utils.compute_outer_scale_exp_trick(time_lags_lr, r_vec_lr, plot=False)

# print(f"Correlation time = {np.round(tc_exp)}s = {np.round(tc_exp/3600)} hours")


# Calculating correlation time, using full 60-day dataset.
# Note highly wiggly ACF due to gaps.
#
# Also calculating $\lambda_C$ using integral method


# fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
# tc, fig, ax = utils.compute_outer_scale_integral(time_lags_lr, r_vec_lr, fig, ax, True)
# # print(f"Correlation time = {np.round(tc)}s = {np.round(tc/3600)} hours")

# plt.show()
# print(tc / 60 / 60)
# print("hours")


# print(
#     "10 of these is ",
#     np.round(10 * tc / 60 / 60 / 24, 2),
#     "days, compared with full data length of",
#     subset.index[-1] - subset.index[0],
# )


# Note that this integral version is much longer, which will make the final interval lengths
# *slightly* more comparable with Fraternale's ints.
# (But note in either case the underlying ACF does look a bit silly)


# ## Apples with oranges: Frat's SFs


# Plot structure functions
xlim = (1e3, 1e7)
ylim_sf = (1e-8, 1e0)
ylim_kurt = (2, 10)

# sdk = sf[[2, 4]].copy()
# sdk["kurtosis"] = sdk[4].div(sdk[2] ** 2)

print("Plotting...")
fig = plt.figure(figsize=(7, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])

ax0.plot(subset_resampled.BT, c="black", lw=0.2)
ax0.set_title(r"Voyager 1 @ 118au, 80\% missing (D1 from Fraternale et al. (2019))")
ax0.set_ylabel("$B_T$ (nT)")
ax0.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax0.xaxis.get_major_locator())
)

for p in [1, 2, 3, 4]:
    ax1.plot(sf_full.index, sf_full[p], c="red")
    qi, xi, pi = utils.fitpowerlaw(sf_full.index, sf_full[p].values, 1e4, 2e5)
    ax1.plot(sf_full_lint.index, sf_full_lint[p], c="purple")

ax1.plot(sf_full.index, sf_full[1], c="red", label=f"Naive SFs")
ax1.plot(sf_full_lint.index, sf_full_lint[1], c="purple", label=f"LINT SFs")

ax1.semilogx()
ax1.semilogy()
ax1.set(title="Fig. 6(d): Structure functions (orders 1-4)", xlabel="Lag (s)")
if xlim is not None:
    ax1.set_xlim(xlim)
if ylim_sf is not None:
    ax1.set_ylim(ylim_sf)

ax1.set_xlabel("$\\tau$ (s)")
ax1.set_ylabel("$S_p$ (nT$^p$)", color="red")
ax1.tick_params(axis="y", labelcolor="red")

rectangle_x = sf_full[sf_full.color == "gray"].index[0]
rectangle_width = (
    sf_full[sf_full.color == "gray"].index[-1]
    - sf_full[sf_full.color == "gray"].index[0]
)

# Add rectangle to show the range of the power-law fit
ax1.add_patch(
    plt.Rectangle(
        (rectangle_x, 1e-8),
        rectangle_width,
        1,
        color="black",
        alpha=0.1,
    ),
)

# Create twin axis BEFORE plotting N(tau)
ax2 = ax1.twinx()

# Plot N(tau) lines with very low zorder
ax2.plot(
    sf_full.index,
    0.2 * (len(subset_resampled) - (sf_full.index / 600)),
    color="black",
    ls="dotted",
    zorder=-1,  # Very low zorder
)
ax2.plot(
    sf_full.index,
    sf_full["N"],
    color="black",
    zorder=-1,  # Very low zorder
    linewidth=1,
)

ax2.set_ylabel("$N(\\tau)$", color="black")
ax2.tick_params(axis="y", labelcolor="black")

# Add vertical lines and text annotations
ax1.axvline(
    x=12 * 3600,
    c="black",
    lw=1.5,
    alpha=0.4,
    label="Multiples of $\\tau=$12h",
    ls="dashed",
)
ax1.axvline(x=24 * 3600, c="black", lw=1.5, alpha=0.3, ls="dashed")
ax1.axvline(x=36 * 3600, c="black", lw=1.5, alpha=0.2, ls="dashed")
ax1.axvline(x=48 * 3600, c="black", lw=1.5, alpha=0.2, ls="dashed")
ax1.axvline(x=60 * 3600, c="black", lw=1.5, alpha=0.2, ls="dashed")
ax1.axvline(x=72 * 3600, c="black", lw=1.5, alpha=0.2, ls="dashed")

ax1.text(rectangle_x, 2e-8, "$N(\\tau)<$ threshold", fontsize=11, alpha=0.8)
ax1.text(1.2e3, 5e-1, "Theoretical trend of $N(\\tau)$", fontsize=11)

# Create legend with higher zorder and solid background
legend = ax1.legend(
    facecolor="white", framealpha=1.0, edgecolor="black", loc="lower left"
)
legend.set_zorder(ax2.get_zorder() + 50)  # Very high zorder
ax1.set_xlim(5e2, 5e6)
# plt.show()
plt.savefig("bg_figs/frat_2019_sfs_reproduction.png")
