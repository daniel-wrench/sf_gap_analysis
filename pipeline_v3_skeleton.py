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
import src.sf_funcs as sf
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

    # Retain the original interval
    original_metadata = metadata.copy()
    original_metadata["gap_status"] = "original"
    modified_intervals.append(original_metadata)

    for j in range(times_to_gap):
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
        gapped_metadata["gap_status"] = "gapped"
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


def get_vector_stats(modified_intervals_list):
    """
    Process all intervals and compile results

    Parameters:
    - modified_intervals_list: list of lists of (DataFrame, metadata) tuples

    Returns:
    - DataFrame with results
    """
    vector_stats_list = []

    for interval_idx, modified_intervals in enumerate(modified_intervals_list):
        for interval_df, metadata in modified_intervals:

            # Interval metadata
            start_time = interval_df.index[0]
            end_time = interval_df.index[-1]
            duration_s = (end_time - start_time).total_seconds()

            # Compute vector stats
            sf, sf_lags, sf_lags_n = compute_sf(interval_df, ["Vx", "Vy", "Vz"])
            acf, acf_lags, sf_lags_n = compute_acf(
                interval_df, ["Vx", "Vy", "Vz"]
            )  # OR, compute from sf
            psd, psd_freq = compute_psd(
                interval_df, ["Vx", "Vy", "Vz"]
            )  # OR, compute from sf

            # Prepare row
            int_results = {
                "spacecraft": metadata["spacecraft"],
                "interval_id": interval_idx,
                "start_time": start_time,
                "duration_s": duration_s,
                "data_points": len(interval_df),
                "gap_status": metadata["gap_status"],
                "tgp": metadata["tgp"],
                "sf": sf,
                "sf_lags": sf_lags,
                "sf_lags_n": sf_lags_n,
                "acf": acf,
                "acf_lags": acf_lags,
                "acf_lags_n": sf_lags_n,
                "psd": psd,
                "psd_freq": psd_freq,
            }

            vector_stats_list.append(int_results)

    # Convert to DataFrame
    return vector_stats_list


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
resample_freq = "1s"

# Resample with mean aggregation
df = df_raw.resample(resample_freq).mean()

# Handle potential NaN values from resampling
df = df.interpolate(method="linear")

# Split into intervals of chosen length
interval_length = "1h"  # 1 hour intervals
clean_intervals = split_into_intervals(df, interval_length, spacecraft)
# Outputs list of dictionaries, including metadata and data for each interval

clean_intervals[0]["data"].plot()

# Add gapped and linear interpolated versions of each interval
all_intervals = []
for interval in clean_intervals:
    gapped = gap_fill_int(metadata=interval, times_to_gap=5)
    all_intervals.append(gapped)

all_intervals[0][0]["data"].plot()
all_intervals[0][1]["data"].plot()
all_intervals[0][2]["data"].plot()

# Compute vector statistics (e.g., SF, ACF, PSD) for each interval
# MAKE SURE WE CAN RUN THIS AND BELOW ON BOTH clean_intervals AND all_intervals
vector_stats_list = get_vector_stats(all_intervals)

# Compute statistics derived from vector stats
for interval in vector_stats_list:
    # Compute derived stats from vector stats
    interval["tce"] = compute_tce(interval["acf"])
    interval["ttu"] = compute_ttc(interval["acf"])
    interval["qi_sf"] = compute_slope(interval["sf"], fit_range)
    interval["qi_psd"] = compute_qi(interval["psd"], fit_range)


vector_stats_df = pd.DataFrame(vector_stats_list)

##################################################

scalar_stats = compute_scalar_stats(vector_stats)
all_scalar_stats.append(scalar_stats)

# Compute means
means = compute_means(["Vx", "Vy", "Vz", "density"])

# Save results
results.to_pickle("analysis_results.pkl")


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
