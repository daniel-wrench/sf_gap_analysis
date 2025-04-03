import glob
import os
import pickle
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sunpy.timeseries import TimeSeries
from sunpy.util import SunpyUserWarning

import src.data_import_funcs as dif
import src.params as params
import src.sf_funcs as sf_funcs
import src.ts_dashboard_utils as ts
import src.utils as utils  # copied directly from Reynolds project, normalize() added


def split_into_intervals(dataframe, interval_length, spacecraft):
    """
    Split dataframe into intervals of specified length

    Parameters:
    - dataframe: pandas DataFrame with timestamp index
    - interval_length: string specifying pandas time offset ('1H', '30min', etc.)

    Returns:
    - List of DataFrames, each containing an interval
    """
    intervals = []
    start_time = dataframe.index[0]
    end_time = dataframe.index[-1]

    current_start = start_time
    int_idx = 0
    while current_start < end_time:
        current_end = current_start + pd.Timedelta(interval_length)
        interval_data = dataframe.loc[current_start:current_end].copy()

        metadata = {
            "spacecraft": spacecraft,
            "interval_id": int_idx,
            "start_time": current_start,
            "end_time": current_end,
            "duration": interval_length,
            "n": len(interval_data),
        }

        # Only include intervals with sufficient data
        if len(interval_data) > 10:  # Minimum threshold
            # Add metadata to the interval
            metadata["data"] = interval_data
            intervals.append(metadata)
            int_idx += 1

        current_start = current_end

    # Print summary of intervals
    print(f"Created {len(intervals)} intervals of length {interval_length}.")
    return intervals


# Each interval in the list has the same structure as df_resampled


def gap_fill_int(metadata, times_to_gap=5):
    """
    Create modified copies of an interval with different data removal patterns

    Parameters:
    - metadata: Dictionary containing interval data and metadata
    - times_to_gap: Number of gapped versions to create

    Returns:
    - List of modified metadata dictionaries
    """
    modified_intervals = []
    minimum_missing_chunks = 0.7

    for j in range(times_to_gap):

        # Retain the original interval
        original_metadata = metadata.copy()
        original_metadata["version"] = j
        original_metadata["gap_status"] = "original"
        original_metadata["tgp"] = np.nan
        original_metadata["data"] = metadata["data"]
        modified_intervals.append(original_metadata)

        # Create a copy of the interval
        interval_df = metadata["data"].copy()

        total_removal = np.random.uniform(0, 0.95)
        ratio_removal = np.random.uniform(minimum_missing_chunks, 1)
        prop_remove_chunks = total_removal * ratio_removal

        data_gapped_chunks, data_gapped_ind_chunks, prop_removed_chunks = (
            ts.remove_data(
                interval_df, prop_remove_chunks, chunks=np.random.randint(1, 10)
            )
        )

        # Calculate amount to remove uniformly
        prop_remove_unif = total_removal - prop_removed_chunks

        # Add the uniform gaps on top of chunks gaps
        data_gapped, data_gapped_ind, prop_removed = ts.remove_data(
            data_gapped_chunks, prop_remove_unif
        )

        # Create and update metadata for gapped version
        gapped_metadata = metadata.copy()
        gapped_metadata["version"] = j
        gapped_metadata["gap_status"] = "naive"
        gapped_metadata["tgp"] = total_removal
        gapped_metadata["data"] = data_gapped
        modified_intervals.append(gapped_metadata)

        # Create interpolated version
        data_lint = data_gapped.interpolate(method="linear").ffill().bfill()

        # Create and update metadata for the linted version
        lint_metadata = metadata.copy()
        lint_metadata["version"] = j
        lint_metadata["gap_status"] = "lint"
        lint_metadata["tgp"] = total_removal
        lint_metadata["data"] = data_lint
        modified_intervals.append(lint_metadata)

    return modified_intervals


def get_vector_stats(interval):
    """
    Process all intervals and compile results

    Parameters:
    - modified_intervals_list: list of lists of (DataFrame, metadata) tuples

    Returns:
    - DataFrame with results
    """

    data = interval["data"]

    lags = np.arange(1, params.max_lag_prop * len(data))
    powers = [2]

    # Compute vector stats
    sf = sf_funcs.compute_sf(data, lags, powers, False, False, None)
    # Bunch of unnecessay columns made here, also don't want
    # option of computing slope, better to do this later
    var = np.sum(np.var(data, axis=0))
    acf_from_sf = 1 - (sf["sf_2"] / (2 * var))

    # acf, acf_lags, sf_lags_n = compute_acf(
    #     interval_df, ["Vx", "Vy", "Vz"]
    # )  # OR, compute from sf
    # psd, psd_freq = compute_psd(
    #     interval_df, ["Vx", "Vy", "Vz"]
    # )  # OR, compute from sf

    # Prepare row
    vector_results = {
        "sf": sf["sf_2"].values,
        "lag": lags,
        "lag_n": sf["n"].values,
        "acf": acf_from_sf.values,
        # "acf_lags": acf_lags,
        # "acf_lags_n": sf_lags_n,
        # "psd": psd,
        # "psd_freq": psd_freq,
    }

    # Convert to DataFrame
    return vector_results


def get_scalars_from_vec(interval):
    """
    Process all intervals and compile results

    Parameters:
    - modified_intervals_list: list of lists of (DataFrame, metadata) tuples

    Returns:
    - DataFrame with results
    """

    # Calculate correlation scale from ACF
    tce = utils.compute_corr_scale_exp_trick(
        interval["lag"],
        interval["acf"],
        plot=False,
    )

    # Calculate Taylor scale from ACF
    ttu, taylor_scale_u_std = utils.compute_taylor_chuychai(
        interval["lag"],
        interval["acf"],
        tau_min=params.tau_min,
        tau_max=params.tau_max,
    )

    # Fit log-log slope to specific range of structure function
    fit_idx = np.where(
        (interval["lag"] >= params.pwrl_range[0])
        & (interval["lag"] <= params.pwrl_range[1])
    )[0]
    qi_sf = np.polyfit(
        np.log(interval["lag"][fit_idx]),
        np.log(interval["sf"][fit_idx]),
        1,
    )[0]

    # Prepare row
    scalar_results = {"tce": tce, "ttu": ttu, "qi_sf": qi_sf}

    # Convert to DataFrame
    return scalar_results


################################################

## PART 1: CALCULATE STATS FOR EACH INTERVAL, PER FILE

# if __name__ == "__main__":

# Read data

data = TimeSeries(
    "data/raw/psp/psp_fld_l2_mag_rtn_2018110200_v02.cdf",
    concatenate=True,
)

df_raw = data.to_dataframe()

spacecraft = "psp"

df_raw = df_raw.loc[:, params.mag_vars_dict[spacecraft]]

# Expected input structure
# timestamp (index) | Bx | By | Bz | Vx | Vy | Vz | density

# Define target frequency (e.g., '1min', '5s')
resample_freq = "10s"

# Resample with mean aggregation
df = df_raw.resample(resample_freq).mean()

# Handle potential NaN values from resampling
df = df.interpolate(method="linear")

# Split into intervals of chosen length
interval_length = "1h"  # 1 hour intervals
clean_intervals = split_into_intervals(df, interval_length, spacecraft)
# Outputs list of dictionaries, including metadata and data for each interval
print(len(clean_intervals))

clean_intervals[0]["data"].plot()
plt.show()

# Add gapped and linear interpolated versions of each interval
times_to_gap = 3
# times_to_gap = params.times_to_gap

all_intervals = []
for interval in clean_intervals:
    gapped = gap_fill_int(metadata=interval, times_to_gap=times_to_gap)
    all_intervals.append(gapped)

print(len(all_intervals))

all_intervals[0][0]["data"].plot()
all_intervals[0][1]["data"].plot()
all_intervals[0][2]["data"].plot()
plt.show()

# Compute vector statistics (e.g., SF, ACF, PSD) for each interval
# and add to metadata

for interval_group in all_intervals:
    for interval in interval_group:
        vector_stats = get_vector_stats(interval)
        interval.update(vector_stats)

# IF NO GAPPING:
# for interval in clean_intervals:
#     vector_stats = get_vector_stats(interval)
#     interval.update(vector_stats)


def plot_intervals_and_stats(index, stat_to_plot, all_intervals, times_to_gap):
    """
    Plot example intervals and corresponding statistics.

    Parameters:
    - index: Index of the interval group to plot (e.g., 0)
    - stat_to_plot: Statistic to plot (e.g., 'acf', 'sf')
    - all_intervals: List of interval groups with metadata and data
    - times_to_gap: Number of gapped versions to plot
    """
    example_clean_int = pd.DataFrame(all_intervals[index])
    # Status colour-mapping
    status_colours = {
        "original": "black",
        "naive": "red",
        "lint": "blue",
    }

    fig, ax = plt.subplots(
        times_to_gap, 2, figsize=(6, times_to_gap * 1.5), sharex="col", sharey="col"
    )
    for version in range(times_to_gap):
        subset = example_clean_int[example_clean_int["version"] == version]

        for status in subset["gap_status"].unique():
            subsubset = subset[subset["gap_status"] == status]
            if status == "original":
                lw = 2.5
            else:
                lw = 0.8
            for i, row in subsubset.iterrows():
                # Plot data
                ax[version, 0].plot(
                    row["data"].iloc[:, 0],
                    label=status,
                    color=status_colours[status],
                    lw=lw,
                )
                # Plot the specified statistic
                ax[version, 1].plot(
                    row["lag"],
                    row[stat_to_plot],
                    label=status,
                    color=status_colours[status],
                    lw=lw,
                )
                if stat_to_plot == "sf":
                    ax[version, 1].set_xscale("log")
                    ax[version, 1].set_yscale("log")

        # Add annotation in the first panel of each row
        ax[version, 0].annotate(
            f"{subset['tgp'].values[1]*100:.2f}% missing",
            xy=(0.4, 0.8),
            xycoords="axes fraction",
            ha="center",
            fontsize=10,
            color="black",
        )

    # Place legend outside the panels
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(status_colours),
    )

    ax[-1, 0].tick_params(axis="x", rotation=45)  # Rotate x-axis tick labels
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend


plot_intervals_and_stats(0, "sf", all_intervals, times_to_gap)

# Compute vector-derived scalar statistics (e.g., tce, ttu, sf_slope) for each interval
# and add to metadata


for interval_group in all_intervals:
    for interval in interval_group:
        scalar_stats = get_scalars_from_vec(interval)
        interval.update(scalar_stats)


# Compute means *of clean intervals, mostly* and add to metadata
for interval in clean_intervals:
    mean_stats = get_mean_stats(interval)
    interval.update(mean_stats)

# SAVE RESULTS:
# - full
# - just scalars


# Create a new version of each dictionary, with the vector values removed
# and the scalar values retained
all_scalar_stats = []
for interval_group in all_intervals:
    for interval in interval_group:
        scalar_stats = {
            "tce": interval["tce"],
            "ttu": interval["ttu"],
            "qi_sf": interval["qi_sf"],
            "qi_psd": interval["qi_psd"],
            "gap_status": interval["gap_status"],
            "tgp": interval["tgp"],
            "version": interval["version"],
        }
        all_scalar_stats.append(scalar_stats)

# Save results
results.to_pickle("analysis_results.pkl")

# PART 1 FINISHED
##################################################

## PART 1A: CALCULATE ERRORS FOR VECTOR STATS, PER FILE
# 3_bin_errors.py
pe = bin_errors(results[sf_2, acf])
pickle.dump(pe)

## PART 1B: FINALISE CORRECTION, USING ERRORS FROM ALL FILES
# 4a_finalise_correction.py
plt.savefig(heatmap)
pickle.dump(correction_lookup)

## PART 1C: APPLY CORRECTION TO ALL FILES
# 5_correct_test_sfs.py

## PART 2: COMBINE ALL STATS INTO ONE FILE, CALCULATE DERIVED SCALARS
df["Re_lt"] = df["tce"] / df["ttc"]

pickle.dump(scalar_and_vector_stats)
pd.to_csv("scalar_stats.csv")

## PART 3: SUMMARISE AND PLOT SCALAR RESULTS
df = pd.read_csv("your_data.csv", parse_dates=["timestamp"])
df = df.set_index("timestamp")

df.describe()
pd.corr(df)
sns.pairplot(df)
plt.savefig("pairplot.png")

## PART 3A: SUMMARISE AND PLOT ERROR RESULTS

## PART 4: CREATE INTERACTIVE DASHBOARD TO EXPLORE VECTORS/TIME SERIES FROM SCALARS
# At least have demo about downloading, reading, and plotting specific time series
