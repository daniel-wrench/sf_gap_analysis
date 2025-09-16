# # Apply correction factor to Voyager data (for real)
#
# Usage: `python correct_plot_voyager_ints.py <spacecraft>`
# Where `<spacecraft>` is either `voyager1` or `voyager2`
# There is then an optional second argument to limit the number of intervals corrected.

# CORRELATION LENGTH = 17 DAYS

import pickle
import sys

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from scipy.interpolate import interp1d

import src.params as params
import src.sf_funcs as sf
import src.utils as utils

# Set sans-serif font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# Parse command line arguments
# if len(sys.argv) < 2 or len(sys.argv) > 3:
#     print("Usage: python correct_plot_voyager_ints.py <spacecraft> [n_intervals]")
#     print("  spacecraft: 'voyager1' or 'voyager2'")
#     print("  n_intervals: optional, limits number of intervals to process")
#     print("Example: python correct_plot_voyager_ints.py voyager1 5")
#     sys.exit(1)

spacecraft = "voyager1"
# spacecraft = sys.argv[1]
if spacecraft not in ["voyager1", "voyager2"]:
    print("Error: spacecraft must be either 'voyager1' or 'voyager2'")
    sys.exit(1)

# Set spacecraft-specific variables
spacecraft_num = "1" if spacecraft == "voyager1" else "2"
spacecraft_short = "v1" if spacecraft == "voyager1" else "v2"
spacecraft_title = "Voyager 1" if spacecraft == "voyager1" else "Voyager 2"

# Constants
CORRELATION_LENGTH_DAYS = 17
TC_N = 10  # Number of correlation lengths per interval
NEW_CADENCE = 288  # 6-pt average, following Frat2021
N_BINS = 25
RUN_MODE = "full"
POWER_LAW_RANGE_FACTOR = [3e3, 3e4]  # Will be divided by cadence
PLOT_DPI = 300


def smooth_scaling(x, y, num_bins=20):
    """Smooth scaling function using logarithmic binning and cubic interpolation."""
    bin_edges = np.logspace(np.log10(x.min()), np.log10(x.max()), num_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    y_binned = np.array(
        [
            y[(x >= bin_edges[i]) & (x < bin_edges[i + 1])].mean()
            for i in range(len(bin_edges) - 1)
        ]
    )

    # Preserve the first and last values to prevent edge distortions
    # during extrapolation
    full_bins = np.insert(bin_centers, 0, bin_edges[0])
    full_bins = np.append(full_bins, bin_edges[-1])
    full_y_binned = np.insert(y_binned, 0, y.iloc[0])
    full_y_binned = np.append(full_y_binned, y.iloc[-1])

    interp_func = interp1d(
        full_bins, full_y_binned, kind="cubic", fill_value="extrapolate"
    )
    return interp_func(x)


# Read in cleaned Voyager data
df = pd.read_pickle(f"data/interim/voyager/{spacecraft}_lism.pkl")
print(f"Loaded {spacecraft} dataset")

# Import lookup table
with open(f"results/{RUN_MODE}/correction_lookup_3d_{N_BINS}_bins_lint.pkl", "rb") as f:
    correction_lookup_3d = pickle.load(f)

# Update lag bins to be relative to a correlation scale
# (noting it was trained on an SF of 2,000 points = 2 corr scales)
correction_lookup_3d["xedges"] = correction_lookup_3d["xedges"] * 10 / params.int_length

# ## Computing standardised SFs
#
# i.e. from intervals of 10,000 points across 10 correlation lengths, calculated up to lag 2,000.
# Integral corr length `tc` has already been defined above.

tc = CORRELATION_LENGTH_DAYS * 24 * 3600  # (17 days in seconds)
new_cadence = NEW_CADENCE

# PREVIOUSLY 1e5, 1e6
pwrl_range = [
    int(POWER_LAW_RANGE_FACTOR[0] / new_cadence),
    int(POWER_LAW_RANGE_FACTOR[1] / new_cadence),
]  # params.pwrl_range
# Reproducing Frat2019 range (5e5,5e6) would require fitting SF up to 60 days

# Previously we chose the cadence based on the # points
# Now we want to choose the number of points based on the cadence

interval_length = int(TC_N * tc / new_cadence)

lags = np.arange(1, params.max_lag_prop * interval_length)
powers = [2]

df_std = df.resample(str(np.round(new_cadence, 3)) + "s").mean()
n_ints = int(np.floor(len(df_std) / interval_length))

# Check if user wants to limit number of intervals
n_ints = 2
# if len(sys.argv) == 3:
#     n_ints_requested = int(sys.argv[2])
#     if n_ints_requested < n_ints:
#         n_ints = n_ints_requested
#         print(f"Processing only the first {n_ints} intervals (user requested).")

print(
    f"Number of standardised intervals to correct: {n_ints} "
    f"({TC_N} corr lengths, {new_cadence}s cadence, {interval_length} points)"
)

# We should have 24 intervals of 10,000 points each, each covering 10 correlation times = 10 * 17 days = 170 days.

del df

# Initialise metadata dataframe
ints_gapped_metadata = pd.DataFrame(
    columns=[
        "file_index",
        "int_index",
        "start_time",
        "end_time",
        "missing",
        "slope",
        "tce",
        "ttu",
        "es_pwr_law_slope",
        "es_pwr_law_coef",
        "es_pwr_law_slope_std",
        "es_pwr_law_coef_std",
    ]
)

file_index = 0
# Just getting all ints from the one dataset for now
# (With consistent resampling)

all_sfs_gapped_corrected = []


# Perform correction for each interval

# ## Smoothing correction
#
# Previous method, employed in first paper submission, involved Gaussian blurring
# the heatmaps to create `correction_lookup_3d_blurred`,
# which replaced `correction_lookup_3d` in the following script.
# *See the GitHub, main branch, for this code.*
#
# This time we are smoothing the actual correction values for each specific SF;
# this is done below and applied to the interval from the paper.
#

COMPONENTS = ["BR", "BT", "BN"]

for int_index in range(n_ints):
    print(f"Correcting interval {int_index}...")
    int_std = df_std[int_index * interval_length : (int_index + 1) * interval_length]

    # sfs_lint = {}
    # sfs_corr = {}
    # sfs_corr_lower = {}
    # sfs_corr_upper = {}

    # for comp in COMPONENTS:
    # sd = np.nanstd(int_std[comp])
    # mean = np.nanmean(int_std[comp])
    # int_std_norm = (int_std[[comp]] - mean) / sd
    # bad_input = pd.DataFrame(int_std_norm)

    int_norm = utils.normalize(int_std)
    bad_input = int_norm[["BR", "BT", "BN"]]

    sd = 1
    bad_output = sf.compute_sf(bad_input, lags, powers, False, False)
    bad_output["gap_handling"] = "naive"
    bad_output["file_index"] = file_index
    bad_output["int_index"] = int_index

    interp_input = (
        bad_input.interpolate(method="linear").ffill().bfill()
    )  # Linearly interpolate (and, in case of missing values at edges, back and forward fill)
    interp_output = sf.compute_sf(interp_input, lags, powers, False, False)

    interp_input_df = pd.DataFrame(interp_input)
    interp_input_df.reset_index(inplace=True)  # Make time a column, not an index

    interp_output["file_index"] = 0
    interp_output["int_index"] = int_index
    interp_output["gap_handling"] = "lint"

    # Correcting sample size and uncertainty for linear interpolation, same values as no handling
    interp_output["n"] = bad_output["n"]
    interp_output["missing_percent"] = bad_output["missing_percent"]
    interp_output["sf_2_se"] = bad_output["sf_2_se"]

    sfs_gapped = pd.concat([interp_output, bad_output])

    # Making lag relative to correlation scale, for consistent correction application
    sfs_gapped["lag_tc"] = sfs_gapped["lag"] * 10 / len(int_std)

    # Apply 2D and 3D scaling to test set, report avg errors
    sfs_lint_corrected_3d = sf.compute_scaling(
        sfs_gapped, 3, correction_lookup_3d, N_BINS
    )

    single_sf = sfs_lint_corrected_3d[(sfs_lint_corrected_3d["int_index"] == int_index)]

    scaling_smooth = smooth_scaling(single_sf.lag, single_sf.scaling)
    scaling_lower_smooth = smooth_scaling(single_sf.lag, single_sf.scaling_lower)
    scaling_upper_smooth = smooth_scaling(single_sf.lag, single_sf.scaling_upper)

    # Save to the main dataframe
    sfs_lint_corrected_3d.loc[
        (sfs_lint_corrected_3d["int_index"] == int_index),
        "scaling_smooth",
    ] = scaling_smooth

    sfs_lint_corrected_3d.loc[
        (sfs_lint_corrected_3d["int_index"] == int_index),
        "scaling_lower_smooth",
    ] = scaling_lower_smooth

    sfs_lint_corrected_3d.loc[
        (sfs_lint_corrected_3d["int_index"] == int_index),
        "scaling_upper_smooth",
    ] = scaling_upper_smooth

    # Apply scalings AND SCALING BACK TO ORIGINAL POWER LEVELS, BASED ON VARIANCE
    sfs_lint_corrected_3d["sf_2_corrected_3d"] = (
        sfs_lint_corrected_3d["sf_2"] * sfs_lint_corrected_3d["scaling_smooth"] * sd**2
    )
    sfs_lint_corrected_3d["sf_2_lower_corrected_3d"] = (
        sfs_lint_corrected_3d["sf_2"]
        * sfs_lint_corrected_3d["scaling_lower_smooth"]
        * sd**2
    )
    sfs_lint_corrected_3d["sf_2_upper_corrected_3d"] = (
        sfs_lint_corrected_3d["sf_2"]
        * sfs_lint_corrected_3d["scaling_upper_smooth"]
        * sd**2
    )

    correction_wide = sfs_lint_corrected_3d[
        [
            "file_index",
            "int_index",
            "lag",
            "missing_percent",
            "sf_2_corrected_3d",
        ]
    ]
    correction_long = pd.wide_to_long(
        correction_wide,
        ["sf_2"],
        i=["file_index", "int_index", "lag", "missing_percent"],
        j="gap_handling",
        sep="_",
        suffix=r"\w+",
    )
    correction_bounds_wide = sfs_lint_corrected_3d[
        [
            "file_index",
            "int_index",
            "lag",
            "missing_percent",
            "sf_2_lower_corrected_3d",
            "sf_2_upper_corrected_3d",
        ]
    ]

    correction_bounds_long = pd.wide_to_long(
        correction_bounds_wide,
        ["sf_2_lower", "sf_2_upper"],
        i=["file_index", "int_index", "lag", "missing_percent"],
        j="gap_handling",
        sep="_",
        suffix=r"\w+",
    )

    corrections_long = pd.merge(
        correction_long,
        correction_bounds_long,
        how="inner",
        on=[
            "file_index",
            "int_index",
            "lag",
            "missing_percent",
            "gap_handling",
        ],
    ).reset_index()

    ########### DO RE-STANDARDISING AND SUMMING OPERATION HERE

    # Adding the corrections, now as a form of "gap_handling", back to the gapped SF dataframe
    sfs_gapped_corrected = pd.concat([sfs_gapped, corrections_long])

    # Calculate slopes and scales
    for gap_handling in sfs_gapped_corrected.gap_handling.unique():

        # Compute equivalent spectrum (ES) for the corrected SFs
        sfs_gapped_corrected.loc[:, "sf_corrected_es"] = (
            sfs_gapped_corrected["sf_2"] * sfs_gapped_corrected["lag"] / 6
        )
        sfs_gapped_corrected.loc[:, "inverse_lag"] = 1 / (sfs_gapped_corrected["lag"])

        # Calculate power-law slope for 2D and 3D corrected SFs
        current_int = sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["gap_handling"] == gap_handling)
        ]

        # Extract power-law fit range of single interval
        fit_range = current_int.loc[
            (current_int["lag"] >= pwrl_range[0])
            & (current_int["lag"] <= pwrl_range[1]),
            :,
        ]

        # Perform the linear regression with full stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log(fit_range["lag"]), np.log(fit_range["sf_2"])
        )
        # sf_corrected_es = current_int["sf_2"] * current_int["lag"] / 6
        # Previously fitted to 100-700 raw lags

        # Get ACF from SF
        # var_signal = np.sum(np.var(input, axis=0))
        var_signal = 3
        # will always be this variance as we are using the standardised 3D SF
        acf_from_sf = 1 - (current_int.sf_2 / (2 * var_signal))
        current_int = current_int.assign(acf_from_sf=acf_from_sf.astype("float32"))

        # Calculate correlation scale from acf_from_sf
        tce = utils.compute_outer_scale_exp_trick(
            current_int["lag"].values,
            current_int["acf_from_sf"].values,
            plot=False,
        )
        # plt.show()
        # NB: if plotting, will not work if tce is not found

        ttu, taylor_scale_u_std = utils.compute_taylor_chuychai(
            current_int["lag"].values,
            current_int["acf_from_sf"].values,
            tau_min=params.tau_min,
            tau_max=params.tau_max,
        )

        # Also change colour when using naive model

        missing = bad_input["BR"].isna().sum() / len(bad_input["BR"])

        # Save results to dataframe

        new_row = pd.DataFrame(
            {
                "file_index": file_index,
                "int_index": int_index,
                "start_time": str(bad_input.index.min()),
                "end_time": str(bad_input.index.max()),
                "cadence": new_cadence,
                "missing": missing,
                "gap_handling": gap_handling,
                "slope": slope,
                "tce": tce,
                "ttu": ttu,
                # "es_pwr_law_slope": popt[1],
                # "es_pwr_law_coef": popt[0],
                # "es_pwr_law_slope_std": pcov[1, 1],
                # "es_pwr_law_coef_std": pcov[0, 0],
            },
            index=[int_index],
        )

        ints_gapped_metadata = pd.concat([ints_gapped_metadata, new_row])

        # Need to add this again for plotting of corrected SF
        sfs_gapped_corrected["lag_tc"] = sfs_gapped_corrected["lag"] * 10 / len(int_std)

    # Append the corrected SFs to the list for later concatenation
    all_sfs_gapped_corrected.append(sfs_gapped_corrected)

    # ##############################################################

    print("Plotting...")

    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(
        2, 2, height_ratios=[1, 1], width_ratios=[1, 1]
    )  # Adjusted to 2 columns with different widths
    gs.update(hspace=0.45, wspace=0.3)

    # First row, spanning both columns
    ax1 = fig.add_subplot(gs[0, :])
    # Second row, two separate columns
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])  # Adjusted to be the second panel

    # Calculate moving average of all bad_input columns
    bad_input_daily = bad_input.resample("1d").mean()

    # Panel 1: Magnetic field plot
    ax1.plot(bad_input.index, bad_input["BR"], lw=0.3, c="red", alpha=0.3)
    ax1.plot(bad_input.index, bad_input["BT"], lw=0.3, c="green", alpha=0.3)
    ax1.plot(bad_input.index, bad_input["BN"], lw=0.3, c="blue", alpha=0.3)
    ax1.plot(
        bad_input_daily.index,
        bad_input_daily["BR"],
        lw=1,
        label=r"$B_R$",
        c="red",
        alpha=0.8,
    )
    ax1.plot(
        bad_input_daily.index,
        bad_input_daily["BT"],
        lw=1,
        label=r"$B_T$",
        c="green",
        alpha=0.8,
    )
    ax1.plot(
        bad_input_daily.index,
        bad_input_daily["BN"],
        lw=1,
        label=r"$B_N$",
        c="blue",
        alpha=0.8,
    )

    ax1.legend(ncol=3, fontsize=10, frameon=True)
    ax1.set_xlabel("Date")
    ax1.set_ylabel(r"$B$ (normalized)")
    ax1.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator())
    )

    # Panel 2: SF plots
    ax2.set_xlabel("Lag (s)")
    ax2.set_ylabel("SF")
    for handling, color, label in zip(
        ["naive", "lint", "corrected_3d"],
        ["indianred", "#7570b3", "black"],
        ["Naive", "LINT", "Corrected"],
    ):
        mask = (
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["gap_handling"] == handling)
            & (sfs_gapped_corrected["lag_tc"] >= 0.00099)
        )
        ax2.plot(
            sfs_gapped_corrected.loc[mask, "lag"] * new_cadence,
            sfs_gapped_corrected.loc[mask, "sf_2"],
            color=color,
            lw=1,
            label=label,
        )

    sf_lags = sfs_gapped_corrected.loc[mask, "lag"]

    # Create smooth x values for the fit line
    sf_lag_fit = np.linspace(pwrl_range[0], pwrl_range[1], 100)
    sf_log_lag_fit = np.log(sf_lag_fit)
    # Calculate prediction bands
    sf_log_y_fit = intercept + slope * sf_log_lag_fit
    # Transform back to original scale
    sf_fit = np.exp(sf_log_y_fit)

    ax2.plot(
        sf_lag_fit * new_cadence,
        sf_fit * 1.5,  # to raise above SF
        # label="Slope = {:.2f}".format(slope),
        ls="dotted",
        lw=1,
        color="black",
        alpha=0.5,
    )
    # Add annotation for slope value
    ax2.annotate(
        f"$\\beta$ = {slope:.2f}",
        xy=(sf_lag_fit[1] * new_cadence, sf_fit[1] * 2.5),
        # xytext=(10, 10),
        # textcoords="offset points",
        fontsize=8,
        alpha=0.6,
        color="black",
        # arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
    )
    # ax2.axvline(
    #     tce * new_cadence,
    #     color="black",
    #     alpha=0.4,
    #     ls="dotted",
    #     label="TCE (see ACF)",
    # )
    ax2.legend(fontsize=8, frameon=True)
    ax2.semilogx()
    ax2.semilogy()

    # Panel 3: ACF from SF
    ax3.plot(
        current_int["lag"] * new_cadence,
        current_int["acf_from_sf"],
        color="black",
        lw=1,
    )
    ax3.set_xlabel("Lag (s)")
    ax3.axhline(1 / np.e, color="black", ls="dotted", alpha=0.6)
    ax3.axvline(
        tce * new_cadence,
        color="black",
        alpha=0.6,
        ls="dotted",
        label=f"$\lambda_C$ = {tce*new_cadence/3600/24:.1f} days",
    )
    ax3.text(
        tce * new_cadence * 1.05,
        0.08,
        f"$\lambda_C$ = {tce*new_cadence/3600/24:.1f} days",
        fontsize=8,
        alpha=0.6,
    )
    # Create an inset to ax3 that highlights the range of params.tau_min and params.tau_max

    axins = inset_axes(ax3, width="30%", height="30%", loc="upper right")
    axins.plot(
        current_int["lag"] * new_cadence,
        current_int["acf_from_sf"],
        color="black",
        lw=1,
    )
    axins.scatter(
        current_int.loc[params.tau_min : params.tau_max, "lag"] * new_cadence,
        current_int.loc[params.tau_min : params.tau_max, "acf_from_sf"],
        color="black",
        marker="x",
        s=3,
        zorder=10,
        # label=r"$\lambda_T$ max lag range",
    )
    axins.set_xlim(0, (params.tau_max + 3) * new_cadence)
    axins.set_ylim(0.9, 1)
    # axins.legend(bbox_to_anchor=(0.95, -0.3), fontsize=6, frameon=False)
    for tick in axins.get_xticklabels():
        tick.set_fontsize(6)
    for tick in axins.get_yticklabels():
        tick.set_fontsize(6)

    # ax3.legend(loc="lower left", fontsize=8, frameon=False)
    ax3.set_ylabel("ACF")

    fig.suptitle(
        f"{spacecraft_title} interval {int_index}: {new_cadence:.0f}s resolution, {missing*100:.1f}% missing",
        y=0.95,
        fontsize=12,
    )

    # fig.text(0.5, 0.47, "SF-DERIVED CURVES", ha="center", fontsize=15)
    # fig.text(0.24, 0.47, "SF CORRECTION", ha="center", fontsize=15)
    axins.text(
        0.5,
        -0.5,
        f"$\lambda_T$={ttu*new_cadence/3600:.1f} hours",
        ha="center",
        va="center",
        transform=axins.transAxes,
        fontsize=8,
        alpha=0.6,
    )

    ax2.set_ylim(1e-1, 1e1)
    ax3.set_ylim(0, 1)
    plt.savefig(
        f"results/full/plots/voyager/{spacecraft_short}_corrected_{int_index}.png",
        dpi=PLOT_DPI,
    )
    plt.close(fig)

# Save metadata
output_file_path = f"results/full/{spacecraft}_corrected_metadata.csv"
ints_gapped_metadata.to_csv(output_file_path, index=False)
print(f"Stats saved to {output_file_path}")

# Export the corrected SFs
sfs_gapped_corrected_all = pd.concat(all_sfs_gapped_corrected, ignore_index=True)
sfs_gapped_corrected_all.to_pickle(f"results/full/{spacecraft}_corrected_sfs.pkl")
