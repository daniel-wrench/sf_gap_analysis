import glob
import os
import pickle
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    print(
        f"Split into {len(intervals)} intervals of length {interval_length}, each with {len(dataframe)} points at {dataframe.index.freqstr} resolution"
    )
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


def plot_intervals_and_stats(index, stat_to_plot, gapped_intervals):
    """
    Plot example intervals and corresponding statistics.

    Parameters:
    - index: Index of the interval group to plot (e.g., 0)
    - stat_to_plot: Statistic to plot (e.g., 'acf', 'sf')
    - all_intervals: List of interval groups with metadata and data
    """
    int_group = pd.DataFrame(all_intervals[index])
    # Status colour-mapping
    status_colours = {
        "original": "black",
        "naive": "red",
        "lint": "blue",
    }

    versions = int_group["version"].nunique()

    fig, ax = plt.subplots(
        versions, 2, figsize=(6, versions * 1.5), sharex="col", sharey="col"
    )
    for version in range(versions):
        subset = int_group[int_group["version"] == version]

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
    plt.show()


def filter_scalar_values(d):
    return {
        k: v
        for k, v in d.items()
        if not isinstance(v, (np.ndarray, pd.DataFrame, pd.Series))
    }


def process_list_of_dicts(data_list):
    """Apply filtering to each dictionary in the list and convert the result into a DataFrame."""
    filtered_list = [filter_scalar_values(d) for d in data_list]
    return pd.DataFrame(filtered_list)


################################################

## PART 1: CALCULATE STATS FOR EACH INTERVAL, PER FILE


def run_pipeline(input_filepath, config):
    """
    Main function to run the pipeline.

    Parameters:
    - input_filepath: Path to the input CDF file
    - output_dir: Directory to save the results
    - config: dictionary with pipeline configuration
    """

    print(f"Processing file: {input_filepath}")

    # Load data
    data = TimeSeries(input_filepath, concatenate=True)
    df_raw = data.to_dataframe()

    # Extract variables of interest
    df_raw = df_raw.loc[:, config["mag_vars"]]
    print("Loaded data with shape:", df_raw.shape)

    # Resample and handle NaN values
    df = df_raw.resample(config["cadence"]).mean()
    df = df.interpolate(method="linear")

    # Split into intervals of chosen length
    intervals = split_into_intervals(df, config["int_length"], config["spacecraft"])

    if config["times_to_gap"] > 0:
        gapped_intervals_nested = []
        print("Gapping intervals {} different ways...".format(config["times_to_gap"]))
        for interval in intervals:
            gapped = gap_fill_int(
                metadata=interval, times_to_gap=config["times_to_gap"]
            )
            gapped_intervals_nested.append(gapped)

        # Convert this list of list of dictionaries into a list of dictionaries
        intervals = [item for sublist in gapped_intervals_nested for item in sublist]
        print(
            f"After making {config['times_to_gap']} gapped versions and handling them in multiple ways, we have {len(intervals)} structure function estimates."
        )

    intervals[0]["data"].plot()
    intervals[1]["data"].plot()
    intervals[2]["data"].plot()
    plt.show()

    # Process each interval and compute statistics
    print("\nComputing statistics for each interval...")
    for interval in intervals:
        # Compute vector statistics (e.g., SF, ACF, PSD)
        vector_stats = get_vector_stats(interval)
        interval.update(vector_stats)
        # Compute vector-derived scalar statistics (e.g., tce, ttu, sf_slope)
        scalar_stats = get_scalars_from_vec(interval)
        interval.update(scalar_stats)
        # Compute means
        data = interval["data"]
        means = {f"mean_{col}": data[col].mean() for col in data.columns}
        interval.update(means)
    print("Done computing statistics.")
    # plot_intervals_and_stats(0, "sf", intervals)

    df_scalars = process_list_of_dicts(intervals)
    print("\nPeek at final dataframe of scalar statistics:\n")
    print(df_scalars.head())

    return intervals, df_scalars


# if __name__ == "__main__":

# Configuration
config = {
    "spacecraft": "psp",
    "mag_vars": [
        "psp_fld_l2_mag_RTN_0",
        "psp_fld_l2_mag_RTN_1",
        "psp_fld_l2_mag_RTN_2",
    ],
    "cadence": "10s",  # Resample frequency
    "int_length": "1h",  # Interval length
    "times_to_gap": 0,  # Number of gapped versions
    "max_lag_prop": 0.2,  # Maximum lag proportion for SF
    # "pwrl_fit_range": [1, 100],  # Range for power-law fit
}


# Read data
data_path_prefix = ""
spacecraft = config["spacecraft"]

raw_file_list = sorted(
    glob.iglob(f"{data_path_prefix}data/raw/{spacecraft}/" + "/*.cdf")
)

file_index = 0  # Change this to process different files

# full_results, scalar_results_df = run_pipeline(raw_file_list[file_index], config)

full_results, scalar_results_df = run_pipeline(raw_file_list[file_index], config)

# Save results
# (JSON might be better for the big output)
# (and test Parquet reading speed when merging scalar dfs later)

scalars_output_file_path = (
    raw_file_list[file_index]
    .replace("raw", "processed")
    .replace(".cdf", "_scalar_stats.csv")
)

scalar_results_df.to_csv(scalars_output_file_path, index=False)
print(f"\nScalar results saved to: {scalars_output_file_path}")

full_output_file_path = (
    raw_file_list[file_index]
    .replace("raw", "processed")
    .replace(".cdf", "_all_stats.pkl")
)
pickle.dump(full_results, open(full_output_file_path, "wb"))
print(f"Full results saved to: {full_output_file_path}")

print("\nPipeline completed successfully!")

#########################################

########################

# PLOT VECTOR STATS FOR DIFFERENT GAP HANDLING METHODS

# # Flatten into long-format DataFrame
# full_results_long = []
# for interval in full_results:
#     for lag, sf_value in zip(interval["lag"], interval["sf"]):  # Unpack SF values
#         full_results_long.append(
#             {
#                 "interval_id": interval["interval_id"],
#                 "version": interval["version"],
#                 "gap_status": interval["gap_status"],
#                 "tgp": interval["tgp"],
#                 "lag": lag,
#                 "sf": sf_value,
#             }
#         )

# full_results_long_df = pd.DataFrame(full_results_long)

# # Plot the different SF estimates for a given interval and version

# # Filter for a given interval and version
# interval_id = 4
# version = 1
# df_filtered = full_results_long_df[
#     (full_results_long_df["interval_id"] == interval_id)
#     & (full_results_long_df["version"] == version)
# ]

# # Plot SFs for different gap-handling methods
# plt.figure(figsize=(8, 5))
# sns.lineplot(data=df_filtered, x="lag", y="sf", hue="gap_status")
# plt.xlabel("Lag")
# plt.ylabel("Structure Function (SF)")
# plt.title(f"SF Estimations for Interval {interval_id}, Version {version}")
# plt.legend(title="Gap Handling Method")
# plt.show()

# TIDY THIS, INCLUDING SAVING
# Then run with config file a la Claude


# PART 1 FINISHED
##################################################

# ## PART 1A: CALCULATE ERRORS FOR VECTOR STATS, PER FILE
# # 3_bin_errors.py
# pe = bin_errors(results[sf_2, acf])
# pickle.dump(pe)

# ## PART 1B: FINALISE CORRECTION, USING ERRORS FROM ALL FILES
# # 4a_finalise_correction.py
# plt.savefig(heatmap)
# pickle.dump(correction_lookup)

# ## PART 1C: APPLY CORRECTION TO ALL FILES
# # 5_correct_test_sfs.py

# ## PART 2: COMBINE ALL STATS INTO ONE FILE, CALCULATE DERIVED SCALARS
# df["Re_lt"] = df["tce"] / df["ttc"]

# pickle.dump(scalar_and_vector_stats)
# pd.to_csv("scalar_stats.csv")

# ## PART 3: SUMMARISE AND PLOT SCALAR RESULTS
# df = pd.read_csv("your_data.csv", parse_dates=["timestamp"])
# df = df.set_index("timestamp")

# df.describe()
# pd.corr(df)
# sns.pairplot(df)
# plt.savefig("pairplot.png")

# ## PART 3A: SUMMARISE AND PLOT ERROR RESULTS

# ## PART 4: CREATE INTERACTIVE DASHBOARD TO EXPLORE VECTORS/TIME SERIES FROM SCALARS
# # At least have demo about downloading, reading, and plotting specific time series
