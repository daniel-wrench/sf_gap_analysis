import glob
import pickle
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sunpy.timeseries import TimeSeries
from sunpy.util import SunpyUserWarning

import src.params as params
import src.sf_funcs as sf_funcs
import src.ts_dashboard_utils as ts
import src.utils as utils  # copied directly from Reynolds project, normalize() added

# Set seed
np.random.seed(42)

# Suppress the specific SunpyUserWarning
warnings.filterwarnings("ignore", category=SunpyUserWarning)
# Suppress the pandas df.sum() warning
warnings.simplefilter(action="ignore", category=FutureWarning)


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


def gap_and_fill_interval(metadata, times_to_gap=5):
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


def get_curves(interval):
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
    # )
    # psd, psd_freq = compute_psd(
    #     interval_df, ["Vx", "Vy", "Vz"]
    # )
    # equiv_spectrum = "Mark's code"

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


def get_derived_stats(interval):
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
    try:
        ttu, taylor_scale_u_std = utils.compute_taylor_chuychai(
            interval["lag"],
            interval["acf"],
            tau_min=params.tau_min,
            tau_max=params.tau_max,
        )
    except ValueError:
        # Handle case where ttu cannot be calculated
        print("Error calculating ttu, setting to NaN. Maybe missing values in ACF?")
        ttu = np.nan
        taylor_scale_u_std = np.nan

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
    scalar_results = {
        "tce": tce,
        "ttu": ttu,
        "qi_sf": qi_sf,
        # level_sf: level_sf, # e.g. 5min level from Burlaga
    }

    # Convert to DataFrame
    return scalar_results


def plot_gapped_curves(results, stat, interval_id, version):
    """
    Plot the specified statistic (e.g., SF) for a given interval and version.

    Parameters:
    - results: List of dictionaries containing interval data and statistics
    - stat: Statistic to plot (e.g., 'sf')
    - interval_id: ID of the interval to filter
    - version: Version of the interval to filter
    """
    # Extract relevant data
    stat_records = []
    time_series_records = []
    for interval in results:
        # Data in long format
        gap_status = interval["gap_status"]
        # Check if the interval matches the specified ID and version
        if interval["interval_id"] == interval_id and interval["version"] == version:
            for lag, stat_value in zip(
                interval["lag"], interval[stat]
            ):  # Unpack stat values
                stat_records.append(
                    {
                        "spacecraft": interval["spacecraft"],
                        "interval_id": interval["interval_id"],
                        "version": interval["version"],
                        "gap_status": interval["gap_status"],
                        "tgp": interval["tgp"],
                        "lag": lag,
                        stat: stat_value,
                    }
                )
            # Store time series
            df_time_series = interval["data"].copy()
            df_time_series["gap_status"] = gap_status  # Tag with gap method
            time_series_records.append(df_time_series)

    results_long_df = pd.DataFrame(stat_records)

    # Convert time series records to DataFrame (still long format)
    df_time_series = pd.concat(
        time_series_records
    )  # Stack different gap-status time series

    # Ensure gap_status order is "original", "lint", "naive"
    gap_status_order = ["original", "lint", "naive"]
    results_long_df["gap_status"] = pd.Categorical(
        results_long_df["gap_status"], categories=gap_status_order, ordered=True
    )
    df_time_series["gap_status"] = pd.Categorical(
        df_time_series["gap_status"], categories=gap_status_order, ordered=True
    )

    tgp = results_long_df["tgp"].unique()[1]
    spacecraft = results_long_df["spacecraft"][0]

    # === PLOTTING ===

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    palette = {"original": "black", "lint": "blue", "naive": "red"}

    # Time Series Plot
    for gap_status, ts_data in df_time_series.groupby("gap_status", observed=False):
        axes[0].plot(
            ts_data.index,
            ts_data[config["mag_vars"][0]],
            label=f"{gap_status}",
            color=palette[gap_status],
        )

    axes[0].set_ylabel("Magnetic Field Component")
    axes[0].set_xlabel("Time")
    # axes[0].legend(title="Gap Handling")
    # axes[0].set_title(f"Time Series for Interval {interval_id}, Version {version}")

    # Stat Plot
    sns.lineplot(
        data=results_long_df,
        x="lag",
        y=stat,
        hue="gap_status",
        ax=axes[1],
        palette=palette,
    )
    axes[1].set_ylabel(stat.upper())
    plt.suptitle(
        f"{stat.upper()} Estimations for {spacecraft.upper()} Interval {interval_id}, Version {version}: {tgp*100:.1f}% removed"
    )
    axes[1].legend(title="Gap Handling Method")
    if stat == "sf":
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
    plt.tight_layout()
    return fig, axes


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
    df = df.interpolate(method="linear").ffill().bfill()

    # Split into intervals of chosen length
    intervals = split_into_intervals(df, config["int_length"], config["spacecraft"])

    if config["times_to_gap"] > 0:
        gapped_intervals_nested = []
        print("Gapping intervals {} different ways...".format(config["times_to_gap"]))
        for interval in intervals:
            gapped = gap_and_fill_interval(
                metadata=interval, times_to_gap=config["times_to_gap"]
            )
            gapped_intervals_nested.append(gapped)

        # Convert this list of list of dictionaries into a list of dictionaries
        intervals = [item for sublist in gapped_intervals_nested for item in sublist]
        print(
            f"After making {config['times_to_gap']} gapped versions and handling them in multiple ways, we have {len(intervals)} structure function estimates."
        )

    # intervals[0]["data"].plot()
    # intervals[1]["data"].plot()
    # intervals[2]["data"].plot()
    # plt.show()

    # Process each interval and compute statistics
    print("\nComputing statistics for each interval...")
    for interval in intervals:
        # Compute vector statistics (e.g., SF, ACF, PSD)
        vector_stats = get_curves(interval)
        interval.update(vector_stats)

        # IN FUTURE: Correct LINT SFs here

        # Compute vector-derived scalar statistics (e.g., tce, ttu, sf_slope)
        scalar_stats = get_derived_stats(interval)
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


################################################

## PART 1: CALCULATE STATS FOR EACH INTERVAL, PER FILE

if __name__ == "__main__":

    # Configuration
    config = {
        "spacecraft": "psp",
        "mag_vars": [
            "psp_fld_l2_mag_RTN_0",
            "psp_fld_l2_mag_RTN_1",
            "psp_fld_l2_mag_RTN_2",
            # "BGSE_0",
            # "BGSE_1",
            # "BGSE_2",
        ],
        "cadence": "10s",  # Resample frequency
        "int_length": "1h",  # Interval length
        "times_to_gap": 5,  # Number of gapped versions
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

# Plot some quick examples of gapped SFs (or other curves!)
if config["times_to_gap"] > 0:
    int_index = 0
    for version in range(2):
        plot_gapped_curves(full_results, "sf", int_index, version)
        plt.savefig(
            raw_file_list[file_index]
            .replace("raw", "processed")
            .replace(".cdf", f"_sf_{int_index}_{version}.png")
        )


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
