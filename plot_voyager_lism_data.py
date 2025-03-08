# ## Plot Voyager LISM data
#
# Takes 3min to read 15 Voyager files (1 year each)


import glob
import os
import re
import sys

import pandas as pd
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries

import src.utils as utils

# So that I can read in the src files while working here in the notebooks/ folder
# NB: does not affect working directory, so still need ../data e.g. for reading data


# Get list of all Voyager 1 files in directory
v1_file_list = glob.glob("data/raw/voyager/voyager1*")
v2_file_list = glob.glob("data/raw/voyager/voyager2*")


# Read in the data
print("Reading in Voyager 1 data...")
v1 = TimeSeries(v1_file_list, concatenate=True)
print("Reading in Voyager 2 data...")
v2 = TimeSeries(v2_file_list, concatenate=True)


# Extract the year from the first file in file_list (ow for some reason timestamp went back to 1993)
v1_start_year = re.search(r"(\d{4})", v1_file_list[0]).group(1)
v2_start_year = re.search(r"(\d{4})", v2_file_list[0]).group(1)

v1_df_raw = v1.to_dataframe()[v1_start_year:]
v2_df_raw = v2.to_dataframe()[v2_start_year:]

# Calculate the cadence of the time series


modal_cadence = v1_df_raw.index.to_series().diff().mode()[0]
modal_cadence


preset_cadence = "24h"
v1_df = v1_df_raw.resample(preset_cadence).mean()
v2_df = v2_df_raw.resample(preset_cadence).mean()

v1_df.info()


v2_df.info()


missing = v2_df.iloc[:, 0].isna().sum() / len(v2_df)
missing


from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# Functions for converting between timestamp, decimal year, and distance
def timestamp_to_decimal_year(timestamp):
    """Convert a pandas timestamp to decimal year"""
    year = timestamp.year
    days_in_year = (
        366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
    )
    day_of_year = timestamp.dayofyear
    decimal_year = year + (day_of_year - 1) / days_in_year
    return decimal_year


def decimal_year_to_distance(decimal_year):
    # Slope and constant are the results of linear fit to following points from FratEA (2021):
    # 2013.36 = 124au
    # 2019.0 = 144au
    distance = 3.546 * decimal_year - 7015.574
    return distance


def distance_to_decimal_year(distance):
    decimal_year = (distance + 7015.574) / 3.546
    return decimal_year


def decimal_year_to_timestamp(decimal_year):
    """Convert decimal year to timestamp"""
    year = int(decimal_year)
    fraction = decimal_year - year
    days_in_year = (
        366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
    )
    day_of_year = int(fraction * days_in_year) + 1
    return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=day_of_year - 1)


def add_distance_axis(ax, num_ticks=5, round_to=10):
    """
    Add a secondary axis with nicely rounded distance values

    Parameters:
    -----------
    ax : matplotlib axis
        The primary axis
    num_ticks : int
        Approximate number of ticks desired
    round_to : int
        Round distance values to multiples of this number
    """
    # Create a secondary x-axis
    ax2 = ax.twiny()

    # Get the date limits from the primary axis
    date_min, date_max = mdates.num2date(ax.get_xlim())

    # Convert to decimal years
    dec_year_min = timestamp_to_decimal_year(pd.Timestamp(date_min))
    dec_year_max = timestamp_to_decimal_year(pd.Timestamp(date_max))

    # Convert to distances
    dist_min = decimal_year_to_distance(dec_year_min)
    dist_max = decimal_year_to_distance(dec_year_max)

    # Create nice, round distance values
    dist_min_rounded = np.ceil(dist_min / round_to) * round_to
    dist_max_rounded = np.floor(dist_max / round_to) * round_to

    # Generate evenly spaced distance ticks
    distance_ticks = np.linspace(dist_min_rounded, dist_max_rounded, num_ticks)

    # Convert distances back to decimal years
    dec_years = [distance_to_decimal_year(dist) for dist in distance_ticks]

    # Convert decimal years to timestamps
    tick_dates = [decimal_year_to_timestamp(dy) for dy in dec_years]

    # Convert timestamps to matplotlib date numbers
    tick_positions = [mdates.date2num(date) for date in tick_dates]

    # Set the tick positions and labels
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"{dist:.0f}" for dist in distance_ticks])

    # Share the limits with the primary axis
    ax2.set_xlim(ax.get_xlim())

    # Set the label for the secondary axis
    ax2.set_xlabel("Distance from Sun (au)")

    return ax2


# Example usage (assuming 'ax' is your existing axis):
# ax2 = add_distance_axis(ax, num_ticks=6, round_to=10)


# Updated function: if doy_end and year_end are not provided, plot a single vertical line.
def create_highlight_region(
    ax,
    name,
    doy_start,
    year_start,
    doy_end=None,
    year_end=None,
    color="grey",
):
    # Compute the base date from the start parameters.
    date = pd.to_datetime(f"{year_start}-01-01") + pd.DateOffset(days=doy_start - 1)
    ylim = ax.get_ylim()
    if doy_end is None or year_end is None:
        # Plot a single vertical line at the specified date.
        ax.axvline(date, color=color, linestyle="--", alpha=0.5, label=name)
        # Place a label near the top of the line.
        ax.text(
            date - pd.DateOffset(days=60),
            ylim[1] * 0.85,
            name,
            rotation=90,
            verticalalignment="top",
            fontsize=10,
        )
    else:
        # Compute the end date.
        date_start = date
        date_end = pd.to_datetime(f"{year_end}-01-01") + pd.DateOffset(days=doy_end - 1)
        # Plot a shaded region between the start and end dates.
        ax.axvspan(date_start, date_end, color=color, alpha=0.2, label=name, lw=0.3)
        # Place the label in the center of the region.
        mid_date = date_start + (date_end - date_start) / 2
        ax.text(
            mid_date,
            ylim[0] + 0.5,
            name,
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="top",
        )


# Load Voyager 1 and 2 DataFrames (v1_df, v2_df) here

# Define key dates
v1_hp_date = pd.to_datetime("2012-08-25")
v2_hp_date = pd.to_datetime("2018-11-05")
v1_end_date = v1_df.index[-1]
v1_time_diff = v1_end_date - v1_hp_date


import matplotlib.pyplot as plt


def add_bulleted_textbox(ax, bullet_points, title=None, alpha=0.8):
    """
    Add a left-justified bulleted text box to the right-most third of a matplotlib axes.
    Ensures the box stays within plot boundaries.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the text box to
    bullet_points : list of str
        List of strings to be displayed as bullet points
    title : str, optional
        Title for the text box
    alpha : float, optional
        Transparency of the text box (0-1)
    """
    # Get axes dimensions
    fig = ax.figure
    bbox = ax.get_position()

    # Calculate box position (right-most third with padding)
    x_start = bbox.x0 + (2 / 3) * bbox.width

    # Create text content with bullet points
    if title:
        text = f"{title}\n\n"
    else:
        text = ""

    text = bullet_points

    # Add the text box with left justification
    props = dict(
        boxstyle="round", facecolor="white", alpha=alpha, edgecolor="gray", pad=0.5
    )

    # Use axes coordinates instead of figure coordinates to stay within bounds
    # This places the text in axes coordinates where 0,0 is bottom left and 1,1 is top right
    ax.text(
        0.6,
        0.45,
        text,
        transform=ax.transAxes,  # Use axes transform instead of figure transform
        ha="left",
        va="center",
        bbox=props,
        fontsize=8,
    )


plt.rcParams["text.usetex"] = True


# Create figure and axes
print("Creating plot...")

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharey=True)
ax1, ax2 = axes

# Plot Voyager 1 data
for component, color, lw in zip(
    ["F1", "BR", "BT", "BN"], ["black", "red", "green", "blue"], [0.9, 0.2, 0.2, 0.2]
):
    ax1.plot(v1_df.index, v1_df[component], label=component, color=color, lw=lw)
ax1.axvline(v1_hp_date, color="k", linestyle="--")
ax1.set_ylabel("B (nT)")


# Add bold capitalized annotation in the top right corner
ax1.annotate(
    "VOYAGER 1",
    xy=(0.99, 0.9),
    xycoords="axes fraction",
    fontsize=16,
    fontweight="bold",
    ha="right",
    va="bottom",
)

# Plot Voyager 2 data
for component, color, lw in zip(
    ["F1", "BR", "BT", "BN"], ["black", "red", "green", "blue"], [0.8, 0.3, 0.3, 0.3]
):
    ax2.plot(v2_df.index, v2_df[component], label=component, color=color, lw=lw)
ax2.axvline(v2_hp_date, color="k", linestyle="--")
# Add bold capitalized annotation in the top right corner
ax2.annotate(
    "VOYAGER 2",
    xy=(0.99, 0.9),
    xycoords="axes fraction",
    fontsize=16,
    fontweight="bold",
    ha="right",
    va="bottom",
)
# Add annotations "HELIOSHEATH", "INTERSTELLAR MEDIUM" pre- and post- heliopause
ax1.annotate(
    "HELIOSHEATH",
    xy=(v1_hp_date - pd.DateOffset(months=20), 0.02),
    xycoords=("data", "axes fraction"),
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="bottom",
    alpha=0.5,
)
ax1.annotate(
    "INTERSTELLAR MEDIUM",
    xy=(v1_hp_date + pd.DateOffset(days=30), 0.02),
    xycoords=("data", "axes fraction"),
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="bottom",
    alpha=0.5,
)

ax2.annotate(
    "HELIOSHEATH",
    xy=(v2_hp_date - pd.DateOffset(months=20), 0.02),
    xycoords=("data", "axes fraction"),
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="bottom",
    alpha=0.5,
)
ax2.annotate(
    "INTERSTELLAR MEDIUM",
    xy=(v2_hp_date + pd.DateOffset(days=30), 0.02),
    xycoords=("data", "axes fraction"),
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="bottom",
    alpha=0.5,
)


ax2.set_ylabel("B (nT)")
ax2.set_xlabel("Date")
ax1.set_xlabel("Date")
ax1.legend(loc="upper left", fontsize=8)

# Set the y-axis upper limit to make room for the annotation
ax2.set_ylim(top=ax2.get_ylim()[1] * 1.2)


# Align the x-axis for both plots
ax1.set_xlim([v1_hp_date + pd.DateOffset(years=-2), v1_hp_date + v1_time_diff])
ax2.set_xlim([v2_hp_date + pd.DateOffset(years=-2), v2_hp_date + v1_time_diff])


# Add highlight regions for Voyager 1
v1_highlight_regions = [
    ("L1", 340, 2012, 130, 2013),
    ("I1/L2", 131, 2013, 176, 2014),
    ("I2/L3", 184, 2014, 131, 2015),
    ("I3/L4", 220, 2015, 294, 2016),
    ("I4", 16, 2017, 272, 2017),
    ("I5", 1, 2018, 365, 2018),
    ("sh1", 335, 2012),
    ("sh2", 236, 2014),
    ("pf1", 346, 2016),
    ("pf2", 147, 2020),
    ("B1", 1, 2020, 146, 2020),
    ("B2", 147, 2020, 365, 2020),
    ("B3", 1, 2021, 365, 2021),
    ("B4", 1, 2022, 365, 2022),
    (r"\textbf{W2", 32, 2021, 39, 2021, "green"),
    (r"\textbf{W1", 280, 2011, 287, 2011, "green"),
]
# Add secondary x-axis in units of AU
ax1_sec = add_distance_axis(ax1)
# ax2_sec = add_distance_axis(ax2) # need to get v2 vals

ylim = ax1.get_ylim()
ax1.text(
    v1_hp_date - pd.DateOffset(days=90),
    ylim[1] * 0.85,
    "$\\textbf{hp}$",
    rotation=90,
    verticalalignment="top",
    fontsize=12,
    weight="bold",
)

ax2.text(
    v2_hp_date - pd.DateOffset(days=90),
    ylim[1] * 0.85,
    "$\\textbf{hp}$",
    rotation=90,
    verticalalignment="top",
    fontsize=12,
    weight="bold",
)

ax1.text("2021-03-01", 0.6, "hump")

for region in v1_highlight_regions:
    create_highlight_region(ax1, *region)

v2_highlight_regions = [("pfa", 120, 2019), ("pfb", 244, 2019), ("sha", 180, 2020)]
for region in v2_highlight_regions:
    create_highlight_region(ax2, *region)


# Add the bulleted text box
bullet_points = "\
$\\textbf{LIT REVIEW OF LISM ANALYSIS}$\n \
VOYAGER 1: \n \
L intervals are from FraternaleEA 2019, where they calculated \n \
    • power spectra (using spectral reconstruction techniques), \n \
    • structure functions (orders 1-4), \n \
    • kurtosis, \n \
    • spectral variance, \n \
    • spectral compressibility \n \n\
I intervals are from FraternaleEA 2021, where they also calculated \n \
    • visual SFs (and sample size functions)\n \
    • visual ACFs\n \
    • correlations scales\n \
    • Taylor scales\n \
\n \
B intervals are from BurlagaEA 2024, where they calculated \n \
    • intermittency (using fits to PDFs)\n \
\n \
VOYAGER 2: \n pfa, pfb, and sha were identified in BurlagaEA 2022. \n No spectral analysis was performed."

add_bulleted_textbox(ax2, bullet_points, title="Literature Review:")

plt.suptitle("Magnetic Field Data from the Local Interstellar Medium", fontsize=16)
plt.tight_layout()
plt.savefig("voyager_lism_data.png")
print("Done")
