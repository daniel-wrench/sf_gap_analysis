# STEP 1. FOR EACH INTERVAL: standardise, duplicate, gap, calculate SFs
# This will be distributed across job arrays on an HPC


# Import dependencies

import glob
import os
import pickle
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.data_import_funcs as dif
import src.params as params
import src.sf_funcs as sf
import src.ts_dashboard_utils as ts
import src.utils as utils  # copied directly from Reynolds project, normalize() added

warnings.simplefilter(action="ignore", category=FutureWarning)

# DELETE FOLLOWING ON HPC
plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# For current Wind importing
sys_arg_dict = {
    # arg1
    "mag_path": params.mag_path,
    "proton_path": params.proton_path,
    "electron_path": params.electron_path,
    # arg2
    "mag_vars": [params.timestamp, params.Bwind_vec],
    "proton_vars": [params.timestamp, params.np, params.Tp],
    "electron_vars": [params.timestamp, params.ne, params.Te],
    # arg3
    "mag_thresh": params.mag_thresh,
    "proton_thresh": params.proton_thresh,
    "electron_thresh": params.electron_thresh,
    # arg4
    "dt_hr": params.dt_hr,
    "int_size": params.int_size,
    # arg5
    "dt_lr": params.dt_lr,
}


# Read in data and split into standardised intervals

# Previously each core read in multiple files at a time. I think it will be better for each core to do one file at a time,
# especially given that each raw file contains sufficient *approximate* correlation lengths for us to then calculate the
# *local* outer scale and create our standardise intervals using that.


times_to_gap = params.times_to_gap
minimum_missing_chunks = 0.7
np.random.seed(123)  # For reproducibility

data_path_prefix = params.data_path_prefix

spacecraft = sys.argv[1]
# Ensure necessary directories exist
os.makedirs(f"{data_path_prefix}data/processed/{spacecraft}", exist_ok=True)
os.makedirs("data/corrections/final", exist_ok=True)
os.makedirs("data/corrections/testing", exist_ok=True)
os.makedirs(f"{data_path_prefix}plots/preprocessing/{spacecraft}", exist_ok=True)
os.makedirs("plots/results/final", exist_ok=True)
os.makedirs("plots/results/testing", exist_ok=True)
os.makedirs("data/corrections", exist_ok=True)


raw_file_list = sorted(
    glob.iglob(f"{data_path_prefix}data/raw/{spacecraft}/" + "/*.cdf")
)
if len(raw_file_list) == 0:
    raise ValueError(
        f"No files found in directory{data_path_prefix}data/raw/{spacecraft}/"
    )

# Selecting one file to read in
file_index = 2

if spacecraft == "psp":
    # takes < 1s/file, ~50 MB mem usage, 2-3 million rows

    psp_raw_cdf = dif.read_cdfs(
        [raw_file_list[file_index]],  # LIMIT HERE!
        {"epoch_mag_RTN": (0), "psp_fld_l2_mag_RTN": (0, 3), "label_RTN": (0, 3)},
    )
    psp_raw = dif.extract_components(
        psp_raw_cdf,
        var_name="psp_fld_l2_mag_RTN",
        label_name="label_RTN",
        time_var="epoch_mag_RTN",
        dim=3,
    )
    df_raw = pd.DataFrame(psp_raw)
    df_raw["Time"] = pd.to_datetime("2000-01-01 12:00") + pd.to_timedelta(
        df_raw["epoch_mag_RTN"], unit="ns"
    )
    df_raw = df_raw.drop(columns="epoch_mag_RTN").set_index("Time")

    # Ensuring observations are in chronological order
    df_raw = df_raw.sort_index()

    # df_wind_hr = pd.read_pickle("data/processed/" + params.mag_path + params.dt_hr + ".pkl")
    df_raw = df_raw.rename(
        columns={
            "B_R": "Bx",
            "B_T": "By",
            "B_N": "Bz",
        }
    )

    # print(df_raw.info())

elif spacecraft == "wind":
    # Takes ~90s/file, 11 MB mem usage, 1 million rows
    print("reading file", raw_file_list[file_index])
    df_raw = utils.pipeline(
        raw_file_list[file_index],
        varlist=sys_arg_dict["mag_vars"],
        thresholds=sys_arg_dict["mag_thresh"],
        cadence=sys_arg_dict["dt_hr"],
    )

    # Ensuring observations are in chronological order
    df_raw = df_raw.sort_index()

    # df_wind_hr = pd.read_pickle("data/processed/" + params.mag_path + params.dt_hr + ".pkl")
    df_raw = df_raw.rename(
        columns={
            # params.Bwind: "Bwind",
            params.Bx: "Bx",
            params.By: "By",
            params.Bz: "Bz",
        }
    )

    # print(df_raw.info())

else:
    raise ValueError("Spacecraft not recognized")


missing = df_raw.iloc[:, 0].isna().sum() / len(df_raw)
# If more than 20% of the data is missing initially, we skip this file (want robust correlation times)
if missing > 0.2:
    print("File missing > 20% data; skipping to next file")
    # Append the name of the file that failed to a file for keeping track of failed files
    with open("failed_files.txt", "a") as f:
        f.write(raw_file_list[file_index] + ": File missing > 20% data\n")

    # Remove this file from the directory
    os.remove(raw_file_list[file_index])
    sys.exit()


# The following chunk gives some metadata - not necessary for the pipeline

### 0PTIONAL CODE ###

# if df_raw.isnull().sum() == 0:
#     print("No missing data")
# else:
#     print(f"{df_raw.isnull().sum()} missing points")
# print("Length of interval: " + str(df_raw.notnull().sum()))
# print("Duration of interval: " + str(df_raw.index[-1] - df_raw.index[0]))
# x = df_raw.values

# # Frequency of measurements
# print("Duration between some adjacent data points:")
# print(df_raw.index[2] - df_raw.index[1])
# print(df_raw.index[3] - df_raw.index[2])
# print(df_raw.index[4] - df_raw.index[3])

# a = df_raw.index[2] - df_raw.index[1]
# x_freq = 1 / (a.microseconds / 1e6)
# print("\nFrequency is {0:.1f} Hz (2dp)".format(x_freq))

# print("Mean = {}".format(np.mean(x)))
# print("Standard deviation = {}\n".format(np.std(x)))

### 0PTIONAL CODE END ###


if spacecraft == "psp":
    tc_approx = 500  # starting-point correlation time, in seconds
    cadence_approx = 0.1  # time resolution (dt) of the data, in seconds
    nlags = 50000  # number of lags to compute the ACF over

elif spacecraft == "wind":
    tc_approx = 2000  # s
    cadence_approx = 1  # s
    nlags = 20000

tc_n = 10  # Number of actual (computed) correlation times we want in our standardised interval...
interval_length = params.int_length  # ...across this many points

df = df_raw.resample(str(cadence_approx) + "S").mean()

# Another data quality check: some are less than their stated duration
if len(df) < nlags:
    print("Dataset too small, skipping to next file")
    # Append the name of the file that failed to a file for keeping track of failed files
    with open("failed_files.txt", "a") as f:
        f.write(
            f"{raw_file_list[file_index]}: File unexpectedly short: {df.index[-1] - df.index[0]}\n"
        )

    # Remove this file from the directory
    os.remove(raw_file_list[file_index])
    sys.exit()

# Delete original dataframes
del df_raw

ints = []
tc_list = []
cadence_list = []

# Significant missing data can result in weird oscillating ACFs, so we need to check for this first
time_lags_lr, r_vec_lr = utils.compute_nd_acf(
    [df.Bx, df.By, df.Bz],
    nlags=nlags,
    plot=False,
)

# Previously used utils.computer_outer_scale_exp_trick()
tc, fig, ax = utils.compute_outer_scale_integral(time_lags_lr, r_vec_lr, plot=True)

output_file_path = (
    raw_file_list[file_index]
    .replace("data/raw", "plots/preprocessing")
    .replace(".cdf", "_acf.pdf")
)
plt.savefig(output_file_path, bbox_inches="tight")
plt.close()

if tc < 0:
    tc = tc_approx
    new_cadence = tc_n * tc / interval_length
    print(
        f"Correlation time (integral method) not found for this interval, setting to 500s (default) -> cadence = {new_cadence}s"
    )

else:
    new_cadence = tc_n * tc / interval_length
    print(
        f"Correlation time (integral method) = {np.round(tc,2)}s -> data resampled to new cadence of {np.round(new_cadence,2)}s, for {tc_n}tc across {interval_length} points"
    )

tc_list.append(tc)
cadence_list.append(new_cadence)

try:
    interval_approx_resampled = df.resample(
        str(np.round(new_cadence, 3)) + "S"
    ).mean()  # Resample to higher frequency

    for i in range(
        0, len(interval_approx_resampled) - interval_length + 1, interval_length
    ):
        interval = interval_approx_resampled.iloc[i : i + interval_length]
        # Check if interval is complete (accept at most 1% missing after resampling)
        if interval.Bx.isnull().sum() / len(interval) < 0.01:
            # Linear interpolate (and, in case of missing values at edges, back and forward fill)
            interval = interval.interpolate(method="linear").ffill().bfill()
            int_norm = utils.normalize(interval)
            ints.append(int_norm)
        else:
            print(">1% missing values in a re-sampled interval; skipping")

except Exception as e:
    print(f"An error occurred: {e}")

if len(ints) == 0:
    print("NO GOOD INTERVALS WITH GIVEN SPECIFICATIONS: not proceeding with analysis")
    # Append the name of the file that failed to a file for keeping track of failed files
    with open("failed_files.txt", "a") as f:
        f.write(
            f"{raw_file_list[file_index]}: Interval has too much missing data after re-sampling and/or incompatible correlation time: {np.round(tc,2)}s\n"
        )
    # Remove this file from the directory
    os.remove(raw_file_list[file_index])

else:
    print(
        "Given this correlation time and data quality, this file yields",
        len(ints),
        "standardised interval/s (see details below)\nThese will be now decimated in",
        times_to_gap,
        "different ways",
    )

    fig, ax1 = plt.subplots(figsize=(3.5, 2))
    # ax2 = ax1.twinx()

    ax1.plot(df["Bx"][: ints[0].index[-1]], color="grey", label="Original")
    # ax1.axvline(df.index[0], c="black", linestyle="dashed")
    # [
    #     ax1.axvline(interval.index[-1], c="black", linestyle="dashed")
    #     for interval in ints
    # ]
    ax1.plot(ints[0]["Bx"], color="black", label="Standardized")
    # ax2.axhline(0, c="black", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Time")
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator())
    )
    ax1.set_ylabel("$B_R$")
    ax1.set_xlim(ints[0].index[0], ints[0].index[-1])
    # ax2 = ax1.twiny()
    # ax2.set_xlabel("Duration ($\lambda_C$)")

    # # Set the secondary x-axis limits to cover the same range as the primary x-axis
    # # ax2.set_xlim(ax1.get_xlim())

    # # # Create tick marks and labels for the secondary x-axis
    # secondary_ticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 11)
    # ax2.set_xticks(secondary_ticks)
    # ax2.set_xticklabels(range(0, 11))

    # Add a vertical dotted line at t1 + tc
    for i in range(11):
        ax1.axvline(
            ints[0].index[0] + pd.Timedelta(tc * i, "s"),
            color="black",
            linestyle="dotted",
            lw=1,
        )

    # Annotate the distance between two vertical lines as lambda_C
    x0 = ints[0].index[0] + pd.Timedelta(tc, "s")
    x1 = ints[0].index[0] + pd.Timedelta(tc * 2, "s")
    ax1.annotate(
        r"$\lambda_C$",
        xy=(x0 + (x1 - x0) / 2, 7.5),  # Midpoint and a little above the plot
        # xytext=(0, 20),
        # textcoords='offset points',
        ha="center",
        va="center",
    )
    ax1.annotate(
        r"$	\longleftrightarrow$",
        xy=(x0 + (x1 - x0) / 2, 6),  # Midpoint and a little above the plot
        # xytext=(0, 20),
        # textcoords='offset points',
        ha="center",
        va="center",
    )
    # Make the y-axis label, ticks and tick labels match the line color.
    # plt.suptitle(
    #     f"Standardised solar wind interval/s from {spacecraft.upper()}, given local conditions",
    #     y=1.1,
    #     fontsize=18,
    # )
    # plt.title(
    #     f"{tc_n}$\lambda_C$ ($\lambda_C=${int(tc)}s) across {interval_length} points, $\langle x \\rangle=0$, $\sigma=1$"
    # )
    ax1.legend(loc="lower left")
    output_file_path = (
        raw_file_list[file_index]
        .replace("data/raw", "plots/preprocessing")
        .replace(".cdf", "_ints_std.pdf")
    )
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()

    files_metadata = pd.DataFrame(
        {
            "file_index": file_index,
            "file_start": df.index[0],
            "file_end": df.index[-1],
            "tc": tc_list,
            "cadence": cadence_list,
        }
    )

    ints_metadata = pd.DataFrame(
        {
            "int_start": [interval.index[0] for interval in ints],
            "int_end": [interval.index[-1] for interval in ints],
        }
    )
    ints_metadata.reset_index(inplace=True)
    ints_metadata.rename(columns={"index": "int_index"}, inplace=True)
    ints_metadata.insert(0, "file_index", file_index)
    print(ints_metadata.head())

    del df  # Clear memory of data no longer needed

    # Analyse intervals (get true SF and slope)

    lags = np.arange(1, params.max_lag_prop * params.int_length)

    # Logarithmically-spaced lags?
    # vals = np.logspace(0, 0.2 * len(ints[0]), 50)
    # lags = np.unique(vals.astype(int))

    powers = [2, 4]
    pwrl_range = params.pwrl_range

    sfs = pd.DataFrame()

    for i, input in enumerate(ints):
        # print(f"\nCore {core} processing standardised interval {i}")
        good_output, slope = sf.compute_sf(
            pd.DataFrame(input), lags, powers, False, False, pwrl_range
        )
        good_output.insert(0, "int_index", i)
        good_output.insert(1, "file_index", file_index)
        sfs = pd.concat([sfs, good_output])
        ints_metadata.loc[ints_metadata["int_index"] == i, "slope"] = slope

    # NON-ESSENTIAL: plot example SF + slope
    check_int = 0
    slope = ints_metadata.loc[ints_metadata["int_index"] == check_int, "slope"].values[
        0
    ]
    timestamp = ints_metadata.loc[ints_metadata["int_index"] == check_int, "int_start"][
        0
    ].strftime(
        "%Y-%m-%d %H:%M:%S"
    )  # doesn't need to be too precise

    plt.plot(
        sfs.loc[sfs["int_index"] == check_int, "lag"],
        sfs.loc[sfs["int_index"] == check_int, "sf_2"],
        label="SF",
    )
    dif.pltpwrl(
        pwrl_range[0],
        0.8,
        pwrl_range[0],
        pwrl_range[1],
        slope,
        lw=2,
        ls="--",
        color="black",
        label=f"Log-log slope: {slope:.3f}",
    )
    # Add vertical line at tc
    tc_lag = interval_length / tc_n
    plt.axvline(
        tc_lag, color="black", linestyle="dotted", lw=1, label="Correlation length"
    )
    plt.semilogx()
    plt.semilogy()
    plt.title(f"$S_2(\\tau)$ for interval beginning {timestamp}")
    plt.legend()
    output_file_path = (
        raw_file_list[file_index]
        .replace("data/raw", "plots/preprocessing")
        .replace(".cdf", "_sf_example.pdf")
    )
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()

    # Duplicate, gap, interpolate, re-analyse intervals

    # Here we gap the original intervals different ways, then calculate SF and corresponding slope for gappy (naive) and
    # interpolated (lint) versions of each of these duplicate intervals.

    index_list = []
    version_list = []
    handling_list = []
    missing_list = []
    missing_chunks_list = []
    slopes_list = []

    sfs_gapped = pd.DataFrame()
    ints_gapped = pd.DataFrame()

    for index in range(len(ints)):
        input = ints[index]

        for j in range(times_to_gap):
            total_removal = np.random.uniform(0, 0.95)
            ratio_removal = np.random.uniform(minimum_missing_chunks, 1)
            # print("Nominal total removal: {0:.1f}%".format(total_removal * 100))
            # print("Nominal ratio: {0:.1f}%".format(ratio_removal * 100))
            prop_remove_chunks = total_removal * ratio_removal

            bad_input_chunks, bad_input_ind_chunks, prop_removed_chunks = (
                ts.remove_data(
                    input, prop_remove_chunks, chunks=np.random.randint(1, 10)
                )
            )
            # Now calculate amount to remove uniformly, given that
            # amount removed in chunks will invariably differ from specified amount
            prop_remove_unif = total_removal - prop_removed_chunks

            # Add the uniform gaps on top of chunks gaps
            bad_input, bad_input_ind, prop_removed = ts.remove_data(
                bad_input_chunks, prop_remove_unif
            )
            # if prop_removed >= 0.95 or prop_removed == 0:
            #     # print(">95% or 0% data removed, skipping")
            #     continue

            bad_output = sf.compute_sf(
                pd.DataFrame(bad_input), lags, powers, False, False
            )
            bad_output["file_index"] = file_index
            bad_output["int_index"] = index
            bad_output["version"] = j
            bad_output["gap_handling"] = "naive"
            sfs_gapped = pd.concat([sfs_gapped, bad_output])

            for handling in ["naive", "lint"]:
                index_list.append(index)
                version_list.append(j)
                missing_list.append(prop_removed * 100)
                missing_chunks_list.append(prop_removed_chunks * 100)

                handling_list.append(handling)

                if handling == "naive":
                    slopes_list.append(slope)
                    # Once we are done with computing the SF, add some metadata to the interval
                    bad_input_df = pd.DataFrame(bad_input).copy(deep=True)
                    # So that we don't overwrite the original, relevant when it comes to linear interpolation
                    bad_input_df.reset_index(inplace=True)
                    bad_input_df["file_index"] = file_index
                    bad_input_df["int_index"] = index
                    bad_input_df["version"] = j
                    bad_input_df["gap_handling"] = handling
                    ints_gapped = pd.concat([ints_gapped, bad_input_df])

                elif handling == "lint":
                    interp_input = (
                        bad_input.interpolate(method="linear").ffill().bfill()
                    )  # Linearly interpolate (and, in case of missing values at edges, back and forward fill)
                    interp_output = sf.compute_sf(
                        pd.DataFrame(interp_input), lags, powers, False, False
                    )

                    # # Once we are done with computing the SF, add some metadata to the interval
                    interp_input_df = pd.DataFrame(interp_input)
                    interp_input_df.reset_index(
                        inplace=True
                    )  # Make time a column, not an index
                    interp_input_df["file_index"] = file_index
                    interp_input_df["int_index"] = index
                    interp_input_df["version"] = j
                    interp_input_df["gap_handling"] = handling
                    ints_gapped = pd.concat([ints_gapped, interp_input_df])

                    interp_output["file_index"] = file_index
                    interp_output["int_index"] = index
                    interp_output["version"] = j
                    interp_output["gap_handling"] = handling

                    # Correcting sample size and uncertainty for linear interpolation, same values as no handling
                    interp_output["n"] = bad_output["n"]
                    interp_output["missing_percent"] = bad_output["missing_percent"]
                    interp_output["sf_2_se"] = bad_output["sf_2_se"]

                    sfs_gapped = pd.concat([sfs_gapped, interp_output])

        # Example plot of gapped intervals
        # if index == 0:
        #     fig, ax = plt.subplots(1, 2)
        #     ax[0].plot(input, label="Original", c="grey")
        #     ax[0].plot(interp_input, label="Linearly interpolated", c="black", ls="--")
        #     ax[0].plot(bad_input, label="Naive", c="black")
        #     ax[1].plot(good_output["lag"], good_output["sf_2"], label="Original")
        #     ax[1].plot(bad_output["lag"], bad_output["sf_2"], label="Naive")
        #     ax[1].plot(
        #         interp_output["lag"],
        #         interp_output["sf_2"],
        #         label="Linearly interpolated",
        #     )
        #     plt.legend()
        #     plt.show()

    ints_gapped_metadata = pd.DataFrame(
        {
            "file_index": file_index,
            "int_index": index_list,
            "version": version_list,
            "missing_percent_overall": missing_list,
            "missing_percent_chunks": missing_chunks_list,
            "gap_handling": handling_list,
        }
    )

    print("Exporting processed dataframes to pickle file\n")

    output_file_path = (
        raw_file_list[file_index].replace("raw", "processed").replace(".cdf", ".pkl")
    )

    with open(output_file_path, "wb") as f:
        pickle.dump(
            {
                "files_metadata": files_metadata,
                "ints_metadata": ints_metadata,
                "ints": ints,
                "ints_gapped_metadata": ints_gapped_metadata,
                "ints_gapped": ints_gapped,
                "sfs": sfs,
                "sfs_gapped": sfs_gapped,
            },
            f,
        )
