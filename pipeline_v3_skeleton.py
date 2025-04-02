def get_vector_stats(modified_intervals_list):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import interpolate, signal

    """
    Process all intervals and compile results

    Parameters:
    - modified_intervals_list: list of lists of (DataFrame, description) tuples

    Returns:
    - DataFrame with results
    """
    results = []

    for interval_idx, modified_intervals in enumerate(modified_intervals_list):
        for interval_df, description in modified_intervals:
            # Interval metadata
            start_time = interval_df.index[0]
            end_time = interval_df.index[-1]
            duration = (end_time - start_time).total_seconds() / 3600  # hours

            # Compute vector stats

            # Compute derived (scalar) stats

            # Compute means for velocity and density
            means = compute_means(interval_df, ["Vx", "Vy", "Vz", "density"])

            # Prepare row
            row = {
                "interval_id": interval_idx,
                "description": description,
                "start_time": start_time,
                "end_time": end_time,
                "duration_hours": duration,
                "data_points": len(interval_df),
                "sf_max": np.max(bz_stats["structure_function"]),
                "acf_min": np.min(bz_stats["autocorrelation"]),
                "psd_integral": np.trapz(bz_stats["psd"], bz_stats["frequencies"]),
            }

            # Add means to the row
            row.update(means)

            # Save structure function, ACF and PSD data separately
            row["sf_data"] = bz_stats["structure_function"]
            row["sf_lags"] = bz_stats["lags"]
            row["acf_data"] = bz_stats["autocorrelation"]
            row["acf_lags"] = bz_stats["lags"]
            row["psd_data"] = bz_stats["psd"]
            row["psd_freq"] = bz_stats["frequencies"]

            results.append(row)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def get_scalar_stats(modified_intervals_list):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import interpolate, signal

    """
    Process all intervals and compile results

    Parameters:
    - modified_intervals_list: list of lists of (DataFrame, description) tuples

    Returns:
    - DataFrame with results
    """
    results = []

    for interval_idx, modified_intervals in enumerate(modified_intervals_list):
        for interval_df, description in modified_intervals:
            # Interval metadata
            start_time = interval_df.index[0]
            end_time = interval_df.index[-1]
            duration = (end_time - start_time).total_seconds() / 3600  # hours

            # Compute vector stats

            # Compute derived (scalar) stats

            # Compute means for velocity and density
            means = compute_means(interval_df, ["Vx", "Vy", "Vz", "density"])

            # Prepare row
            row = {
                "interval_id": interval_idx,
                "description": description,
                "start_time": start_time,
                "end_time": end_time,
                "duration_hours": duration,
                "data_points": len(interval_df),
            }

            # Add means to the row
            row.update(means)

            results.append(row)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


# MERGE THE TWO

## PART 1: CALCULATE STATS FOR EACH INTERVAL, PER FILE

if __name__ == "__main__":

    import params
    import utils

    # Read data
    mag_data = read_cdf("mag_data.cdf")

    # Expected input structure
    # timestamp (index) | Bx | By | Bz | Vx | Vy | Vz | density

    # Define target frequency (e.g., '1min', '5s')
    resample_freq = "1min"

    # Resample with mean aggregation
    df_resampled = df.resample(resample_freq).mean()

    # Handle potential NaN values from resampling
    df_resampled = df_resampled.interpolate(method="linear")

    # Split into intervals of chosen length
    interval_length = "1H"  # 1 hour intervals
    intervals = split_into_intervals(df_resampled, interval_length)

    # Each interval in the list has the same structure as df_resampled

    removal_patterns = [
        {"method": "random", "fraction": 0.2},
        {"method": "continuous", "fraction": 0.3},
    ]

    all_modified_intervals = []

    for interval in intervals:
        modified = create_modified_intervals(interval, removal_patterns)
        all_modified_intervals.append(modified)

    all_vector_stats = []
    all_scalar_stats = []

    for interval in all_modified_intervals:

        # Calculate vector statistics
        vector_stats = compute_vector_stats(interval)
        all_vector_stats.append(vector_stats)

        # Calculate scalar statistics
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
