# # Apply correction factor to Voyager data (for real)
#
# CORRELATION LENGTH = 17 DAYS

import pickle

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

# Fit a power law to the corrected SF
from scipy.optimize import curve_fit

import src.params as params
import src.sf_funcs as sf
import src.utils as utils

# Set matplotlib font size
plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# Read in cleaned Voyager 1 data
df = pd.read_pickle("data/interim/voyager/voyager1_lism_cleaned.pkl")
print("Loaded dataset")

# Importing lookup table
n_bins = 25
run_mode = "full"
with open(f"results/{run_mode}/correction_lookup_3d_{n_bins}_bins_lint.pkl", "rb") as f:
    correction_lookup_3d = pickle.load(f)

# Update lag bins to be relative to a correlation scale
# (noting it was trained on an SF of 2,000 points = 2 corr scales)
correction_lookup_3d["xedges"] = correction_lookup_3d["xedges"] * 10 / params.int_length

# ## Computing standardised SFs
#
# i.e. from intervals of 10,000 points across 10 correlation lengths, calculated up to lag 2,000.
# Integral corr length `tc` has already been defined above.


tc = 17 * 24 * 3600  # (17 days in seconds)

tc_n = 10
# interval_length = params.int_length
# new_cadence = tc_n * tc / interval_length
new_cadence = (
    288  # 6-pt average, following Frat2021 while still making high-res enough for SFs
)

pwrl_range = [int(1e5 / new_cadence), int(1e6 / new_cadence)]  # params.pwrl_range
# Reproducing Frat2019 range (5e5,5e6) would require fitting SF up to 60 days

# Previously we chose the cadence based on the # points
# Now we want to choose the number of points based on the cadence

interval_length = int(tc_n * tc / new_cadence)

lags = np.arange(1, params.max_lag_prop * interval_length)
powers = [2]

df_std = df.resample(str(np.round(new_cadence, 3)) + "s").mean()
n_ints = int(np.floor(len(df_std) / interval_length))
print(
    f"Number of standardised intervals to correct: {n_ints} ({tc_n} corr lengths, {new_cadence}s cadence, {interval_length} points)"
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

# Extract an interval
for int_index in range(3):
    print(f"Correcting interval {int_index}...")
    int_std = df_std[int_index * interval_length : (int_index + 1) * interval_length]

    int_norm = utils.normalize(int_std)
    bad_input = int_norm[["BR", "BT", "BN"]]

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

    # ### Correcting SF

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

    # Apply 2D and 3D scaling to test set, report avg errors

    sfs_lint_corrected_3d = sf.compute_scaling(
        sfs_gapped, 3, correction_lookup_3d, n_bins
    )

    from scipy.interpolate import interp1d

    # Smoothing function
    def smooth_scaling(x, y, num_bins=20):
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

    # Apply scalings
    sfs_lint_corrected_3d["sf_2_corrected_3d"] = (
        sfs_lint_corrected_3d["sf_2"] * sfs_lint_corrected_3d["scaling_smooth"]
    )
    sfs_lint_corrected_3d["sf_2_lower_corrected_3d"] = (
        sfs_lint_corrected_3d["sf_2"] * sfs_lint_corrected_3d["scaling_lower_smooth"]
    )
    sfs_lint_corrected_3d["sf_2_upper_corrected_3d"] = (
        sfs_lint_corrected_3d["sf_2"] * sfs_lint_corrected_3d["scaling_upper_smooth"]
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

    # Adding the corrections, now as a form of "gap_handling", back to the gapped SF dataframe
    sfs_gapped_corrected = pd.concat([sfs_gapped, corrections_long])

    # Calculate slopes and scales

    gap_handling = "corrected_3d"

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
        (current_int["lag"] >= pwrl_range[0]) & (current_int["lag"] <= pwrl_range[1]), :
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

    # ##############################################################

    print("Plotting...")

    fig = plt.figure(figsize=(11, 5))
    gs = gridspec.GridSpec(
        2, 3, height_ratios=[1, 1]
    )  # Now 3 columns in the second row
    gs.update(hspace=0.6, wspace=0.3)

    # First row, spanning all three columns
    ax1 = fig.add_subplot(gs[0, :])
    # Second row, three separate columns
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])  # New third panel

    # Panel 1: Magnetic field plot
    ax1.plot(bad_input.index, bad_input["BR"], color="black", lw=0.3, label="Raw")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(r"$B_R$ (normalized)")
    ax1.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator())
    )

    # Panel 2: SF plots
    ax2.set_xlabel("Lag (s)")
    ax2.set_ylabel("SF")
    for handling, color, label in zip(
        ["naive", "lint", "corrected_3d"],
        ["red", "black", "#1b9e77"],
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
        label="Slope = {:.2f}".format(slope),
        ls="dotted",
        lw=1,
        color="#1b9e77",
    )
    ax2.axvline(
        tce * new_cadence,
        color="black",
        alpha=0.4,
        ls="dotted",
        label="TCE (see ACF)",
    )
    ax2.legend(loc="lower right", fontsize=8, frameon=False)
    ax2.semilogx()
    ax2.semilogy()

    # Panel 3: Equivalent spectrum
    ax3.plot(
        current_int["inverse_lag"] / new_cadence,  # Dividing coz inverse lag
        current_int["sf_corrected_es"],
        c="#1b9e77",
    )
    # ax3.plot(
    #     sf_lag_fit * new_cadence,
    #     sf_fit * 10,  # to raise above SF
    #     label="Slope = {:.2f}".format(slope),
    #     ls="dotted",
    #     lw=2.5,
    #     color="#1b9e77",
    # )
    ax3.semilogx()
    ax3.semilogy()
    ax3.legend(loc="lower left", frameon=False)
    ax3.set_xlabel("1/Lag ($s^{-1}$)")
    ax3.set_ylabel(r"$\frac{1}{6} \tau$ SF")

    # Panel 4: ACF from SF
    ax4.plot(
        current_int["lag"] * new_cadence,
        current_int["acf_from_sf"],
        color="#1b9e77",
        lw=1,
    )
    ax4.set_xlabel("Lag (s)")
    ax4.axhline(1 / np.e, color="black", ls="dotted")
    ax4.axvline(
        tce * new_cadence,
        color="black",
        ls="dotted",
        label=f"TCE = {tce*new_cadence/3600/24:.1f} days",
    )
    # Create an inset to ax4 that highlights the range of params.tau_min and params.tau_max

    axins = inset_axes(ax4, width="30%", height="30%", loc="upper right")
    axins.plot(
        current_int["lag"] * new_cadence,
        current_int["acf_from_sf"],
        color="#1b9e77",
        lw=1,
    )
    axins.scatter(
        current_int.loc[params.tau_min : params.tau_max, "lag"] * new_cadence,
        current_int.loc[params.tau_min : params.tau_max, "acf_from_sf"],
        color="black",
        marker="x",
        # REDUCE SIZE
        s=3,
        # BRING TO FRONT
        zorder=10,
        label=r"$\lambda_T$ max lag range",
    )
    axins.set_xlim(0, (params.tau_max + 3) * new_cadence)
    axins.set_ylim(0.9, 1)
    axins.legend(bbox_to_anchor=(0.95, -0.3), fontsize=6, frameon=False)
    # Reduce font size of ticklabels
    for tick in axins.get_xticklabels():
        tick.set_fontsize(6)
    for tick in axins.get_yticklabels():
        tick.set_fontsize(6)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])

    ax4.legend(loc="lower left", fontsize=8, frameon=False)

    fig.suptitle(
        f"Voyager 1 LISM, interval {int_index}: {new_cadence/60:.1f}min resolution, {missing*100:.1f}\% missing",
        y=0.95,
        fontsize=20,
    )

    # Add a shared title above the 3rd and 4th panels
    fig.text(0.65, 0.47, "SF-DERIVED CURVES", ha="center", fontsize=15)
    fig.text(0.24, 0.47, "SF CORRECTION", ha="center", fontsize=15)
    axins.text(
        0.5,
        0.85,
        f"$\lambda_T$={ttu*new_cadence/3600:.1f} hours",
        ha="center",
        va="center",
        transform=axins.transAxes,
        fontsize=6,
    )

    ax2.set_title("SF")
    ax3.set_title("Equivalent Spectrum")
    ax4.set_title("ACF")

    ax1.set_ylim(-3, 3)
    ax2.set_ylim(1e-1, 1e1)
    ax3.set_ylim(1e-2, 1e4)
    ax4.set_ylim(0, 1)
    plt.savefig(f"results/full/plots/voyager/v1_corrected_{int_index}.png", dpi=300)
    plt.close(fig)


# Save metadata
# output_file_path = "results/full/voyager1_corrected_metadata.csv"
# ints_gapped_metadata.to_csv(output_file_path, index=False)
# print(f"Stats saved to {output_file_path}")
