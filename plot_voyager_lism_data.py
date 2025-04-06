## Plot Voyager LISM data

# TO-DO
# - Fix annotations (maybe don't highlight intervals?)
# - Get rid of un-used functions
# - Reduce alpha of pre-heliopause data
# - Add uncertainties

# Takes 3min to read 15 Voyager files (1 year each)

import glob
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sunpy.timeseries import TimeSeries

plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# So that I can read in the src files while working here in the notebooks/ folder
# NB: does not affect working directory, so still need ../data e.g. for reading data


# Get the start and end dates of the first and last non-missing rows
def clean_empty_start_end(df, column_name):
    """
    Returns the index of the first and last non-missing value in a specified column.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze
    column_name (str): The name of the column to check

    Returns:
    tuple: (first_index, last_index) - indices of first and last non-missing values
           Returns (None, None) if all values are missing
    """
    # Get boolean series of non-missing values
    non_missing = df[column_name].notna()

    # If all values are missing, return None for both
    if not non_missing.any():
        return None, None

    # Get index of first non-missing value
    first_index = non_missing.idxmax()

    # Get index of last non-missing value
    # We can reverse the series and find the first True value
    last_index = non_missing[::-1].idxmax()

    df_cleaned = df.loc[first_index:last_index]

    return df_cleaned


# Get list of all files in directory
v1_file_list = glob.glob("data/raw/voyager/voyager1*")
v2_file_list = glob.glob("data/raw/voyager/voyager2*")


# Read in the data
print("Reading in Voyager 1 data...")
v1 = TimeSeries(v1_file_list, concatenate=True)
print("Reading in Voyager 2 data...")
v2 = TimeSeries(v2_file_list, concatenate=True)


# Extract the year from the first file in file_list
# (ow for some reason timestamp went back to 1993)

v1_start_year = re.search(r"(\d{4})", v1_file_list[0]).group(1)
v2_start_year = re.search(r"(\d{4})", v2_file_list[0]).group(1)

v1_df_raw = v1.to_dataframe()[v1_start_year:]
v2_df_raw = v2.to_dataframe()[v2_start_year:]

# Calculate the cadence of the time series


modal_cadence = v2_df_raw.index.to_series().diff().mode()[0]
modal_cadence

preset_cadence = "24h"
v1_df = v1_df_raw.resample(preset_cadence).mean()
v2_df = v2_df_raw.resample(preset_cadence).mean()

v1_df = clean_empty_start_end(v1_df, "F1")
v2_df = clean_empty_start_end(v2_df, "F1")

v2_df.info()

# Get missingness of each column
print(f"V1 missingness since {v1_start_year} at {preset_cadence} cadence:")
v1_df.isna().sum() / len(v1_df) * 100

print(f"V2 missingness since {v2_start_year} at {preset_cadence} cadence:")
v2_df.isna().sum() / len(v2_df) * 100

v1_df.reset_index(names="Date", inplace=True)
v2_df.reset_index(names="Date", inplace=True)

# Using the Radius and Datetime columns, create function to convert from datetime to radius

# Convert datetime index to numeric values for the clean dataframe
datetime_numeric_v2 = mdates.date2num(v2_df["Date"])

# Create an interpolation function using only valid data points
rad_interp_v2 = interp1d(
    datetime_numeric_v2,
    v2_df["Radius"],
    bounds_error=False,
    fill_value="extrapolate",
)

v2_hp_date = pd.to_datetime("2018-11-05")
v2_hp_radius = rad_interp_v2(mdates.date2num(v2_hp_date))
v2_hp_radius

# Convert datetime index to numeric values for the clean dataframe
datetime_numeric_v1 = mdates.date2num(v1_df["Date"])

# Create an interpolation function using only valid data points
rad_interp_v1 = interp1d(
    datetime_numeric_v1,
    v1_df["Radius"],
    bounds_error=False,
    fill_value="extrapolate",
)

v1_hp_date = pd.to_datetime("2012-08-25")
v1_hp_radius = rad_interp_v1(mdates.date2num(v1_hp_date))


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
    # Slope and constant are the results of linear fit to following points (FratEA2021):
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


# Updated function: if doy_end and year_end are not provided,
# plot a single vertical line.
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
    ax1.plot(v1_df.Radius, v1_df[component], label=component, color=color, lw=lw)
v1_hp_radius = rad_interp_v1(mdates.date2num(v1_hp_date))
ax1.axvline(v1_hp_radius, color="k", linestyle="--")
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
    ax2.plot(v2_df.Radius, v2_df[component], label=component, color=color, lw=lw)
v2_hp_radius = rad_interp_v2(mdates.date2num(v2_hp_date))
ax2.axvline(v2_hp_radius, color="k", linestyle="--")
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
# ax1.annotate(
#     "HELIOSHEATH",
#     xy=(v1_hp_date - pd.DateOffset(months=20), 0.02),
#     xycoords=("data", "axes fraction"),
#     fontsize=10,
#     fontweight="bold",
#     ha="left",
#     va="bottom",
#     alpha=0.5,
# )
# ax1.annotate(
#     "INTERSTELLAR MEDIUM",
#     xy=(v1_hp_date + pd.DateOffset(days=30), 0.02),
#     xycoords=("data", "axes fraction"),
#     fontsize=10,
#     fontweight="bold",
#     ha="left",
#     va="bottom",
#     alpha=0.5,
# )

# ax2.annotate(
#     "HELIOSHEATH",
#     xy=(v2_hp_date - pd.DateOffset(months=20), 0.02),
#     xycoords=("data", "axes fraction"),
#     fontsize=10,
#     fontweight="bold",
#     ha="left",
#     va="bottom",
#     alpha=0.5,
# )
# ax2.annotate(
#     "INTERSTELLAR MEDIUM",
#     xy=(v2_hp_date + pd.DateOffset(days=30), 0.02),
#     xycoords=("data", "axes fraction"),
#     fontsize=10,
#     fontweight="bold",
#     ha="left",
#     va="bottom",
#     alpha=0.5,
# )


ax2.set_ylabel("B (nT)")
ax2.set_xlabel("Radial distance from Sun (AU)")
ax1.set_xlabel("Radial distance from Sun (AU)")
ax1.legend(loc="upper left", fontsize=8)

# Set the y-axis upper limit to make room for the annotation
ax2.set_ylim(top=ax2.get_ylim()[1] * 1.2)


# # Align the x-axis for both plots
# ax1.set_xlim([v1_hp_date + pd.DateOffset(years=-2), v1_hp_date + v1_time_diff])
# ax2.set_xlim([v2_hp_date + pd.DateOffset(years=-2), v2_hp_date + v1_time_diff])


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


# Set up top x-axis (dates) as a secondary axis
ax2_top = ax2.twiny()
ax2_top.set_xlim(ax2.get_xlim())

# Find indices where the date is Jan 1st of each year
years = pd.to_datetime(v2_df["Date"])
year_starts = years[years.dt.is_year_start]

# Map those dates to radius positions
radius_year_ticks = year_starts.index

# Set those as ticks on the bottom axis
ax2_top.set_xticks(radius_year_ticks)
ax2_top.set_xticklabels([d.strftime("%Y") for d in year_starts])
ax2_top.xaxis.set_ticks_position("top")
ax2_top.xaxis.set_label_position("top")
ax2_top.spines["top"].set_position(("outward", 0))
ax2_top.set_xlabel("Date")

# Do the same for the top axis of Voyager 1

# Set up top x-axis (dates) as a secondary axis
ax1_top = ax1.twiny()
ax1_top.set_xlim(ax1.get_xlim())

# Find indices where the date is Jan 1st of each year
years = pd.to_datetime(v1_df["Date"])
year_starts = years[years.dt.is_year_start]

# Map those dates to radius positions
radius_year_ticks = year_starts.index

# Set those as ticks on the bottom axis
ax1_top.set_xticks(radius_year_ticks)
ax1_top.set_xticklabels([d.strftime("%Y") for d in year_starts])
ax1_top.xaxis.set_ticks_position("top")
ax1_top.xaxis.set_label_position("top")
ax1_top.spines["top"].set_position(("outward", 0))
ax1_top.set_xlabel("Date")


# ylim = ax1.get_ylim()
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
VOYAGER 2: \n pfa, pfb, and sha were identified in BurlagaEA 2022. \n \
    No spectral analysis was performed."

add_bulleted_textbox(ax2, bullet_points, title="Literature Review:")

plt.suptitle("Magnetic Field Data from the Local Interstellar Medium", fontsize=16)
plt.tight_layout()
plt.savefig("voyager_lism_data.png")
print("Done")
