import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Get Line2d
from matplotlib.lines import Line2D

# Set matplotlib to move all tickmarks inside
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
# Set font to Arial
plt.rcParams["font.family"] = "Arial"


# Read a pickle file
def read_pickle_file(file_path):
    """
    Read a pickle file and return the data.

    Parameters:
    -----------
    file_path : str
        Path to the pickle file.

    Returns:
    --------
    data : object
        Data loaded from the pickle file.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_voyager_analysis_plot(
    v1_res,
    v2_res,
    v1_sfs_corrected,
    v2_sfs_corrected,
    lags_v1_hr,
    sf_v1_hr,
    lags_v2_hr,
    sf_v2_hr,
    es_v1_f,
    es_v1_hr,
    es_v2_f,
    es_v2_hr,
    lc_corr_min,
    lc_corr_max,
):
    """
    Create a comprehensive three-panel plot for Voyager magnetic field analysis.

    Parameters:
    -----------
    v1_res, v2_res : pandas.DataFrame
        Voyager 1 and 2 magnetic field data with 'F1' column
    v1_sfs_corrected, v2_sfs_corrected : pandas.DataFrame
        Corrected structure function data
    lags_v1_hr, sf_v1_hr, lags_v2_hr, sf_v2_hr : array-like
        Structure function lags and values
    es_v1_f, es_v1_hr, es_v2_f, es_v2_hr : array-like
        Equivalent spectra frequencies and values
    """

    # Constants
    U = 30000  # m/s relative velocity ISM wrt spacecraft: U_rel = U_ISM - U_spacecraft = ~13 km/s - -17km/s (Fraternale 2021)
    AU = 1.496e11  # meters (1 astronomical unit)
    DI = 700000  # Ion inertial length in km

    # Create figure with optimized layout
    fig, axes = plt.subplots(3, 1, figsize=(4, 9))

    # Color scheme
    colors = {
        "v1": "#16702df8",
        "v2": "#43a1cabe",
        "reference": "#696969",  # Dim gray
        "k41": "#708090",  # Slate gray
    }

    # =============================================================================
    # Panel 1: Magnetic Field Time Series
    # =============================================================================
    ax1 = axes[0]

    # Plot daily averages with transparency
    v1_res["F1"].plot(
        ax=ax1,
        label="Voyager 1 daily avg",
        color=colors["v1"],
        linewidth=0.8,
        alpha=0.4,
    )
    v2_res["F1"].plot(
        ax=ax1,
        label="Voyager 2 daily avg",
        color=colors["v2"],
        linewidth=0.8,
        alpha=0.4,
    )

    # Add 2-week moving averages
    v1_res["F1"].rolling(window=14, center=True).mean().plot(
        ax=ax1, label="V1 2-week moving avg", color=colors["v1"], linewidth=1.2
    )
    v2_res["F1"].rolling(window=14, center=True).mean().plot(
        ax=ax1, label="V2 2-week moving avg", color=colors["v2"], linewidth=1.2
    )

    # Formatting
    ax1.set_ylabel("|B| (nT)", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)

    ax1.set_ylim(0.32, 0.88)
    ax1.grid(True, alpha=0.3)

    # Add mission information
    _add_mission_info(ax1, colors)

    # =============================================================================
    # Panel 2: Structure Functions
    # =============================================================================
    ax2 = axes[1]

    # Main structure functions
    ax2.loglog(lags_v1_hr * 48, sf_v1_hr, color=colors["v1"], label="V1", linewidth=2)
    ax2.loglog(lags_v2_hr * 48, sf_v2_hr, color=colors["v2"], label="V2", linewidth=2)

    # Corrected structure functions
    _plot_corrected_sfs(ax2, v1_sfs_corrected, colors["v1"], scale_factor=3000)
    _plot_corrected_sfs(ax2, v2_sfs_corrected, colors["v2"], scale_factor=3000)

    # Kolmogorov reference lines
    _add_k41_references_sf(ax2, colors["reference"])

    # Formatting
    ax2.set_xlabel("$\\tau$ (s)", fontsize=12)
    ax2.set_ylabel("$S_2$ (nT$^2$)", fontsize=12)
    ax2.set_ylim(1e-6, 5e-1)
    ax2.set_xlim(1e1, 3e8)
    ax2.grid(True, alpha=0.3)
    # ax2.legend(fontsize=10)

    # Add secondary axis for spatial scales
    ax2_spatial = _add_spatial_axis(ax2, U)

    # Add time and length scale annotations
    _add_scale_annotations(ax2, ax2_spatial)

    # =============================================================================
    # Panel 3: Equivalent Spectra
    # =============================================================================
    ax3 = axes[2]

    # Main equivalent spectra
    ax3.loglog(es_v1_f, es_v1_hr, label="V1", color=colors["v1"], linewidth=2)
    ax3.loglog(es_v2_f, es_v2_hr, label="V2", color=colors["v2"], linewidth=2)

    # Corrected equivalent spectra
    _plot_corrected_es(ax3, v1_sfs_corrected, colors["v1"])
    _plot_corrected_es(ax3, v2_sfs_corrected, colors["v2"], scale_factor=10)

    # Kolmogorov reference lines
    _add_k41_references_es(ax3, es_v1_f, colors["reference"])

    # Formatting
    ax3.set_xlabel("f (Hz)", fontsize=12)
    ax3.set_ylabel("$S_2\\tau/6$", fontsize=12)
    ax3.set_ylim(7e-5, 1e7)
    ax3.grid(True, alpha=0.3)
    # ax3.legend(fontsize=10)

    # Add secondary axis for wavenumber
    _add_wavenumber_axis(ax3, U)

    # Annotate fit range
    x1, x2 = 1e3, 1e4  # your x-values
    y_pos = 2e-2  # y-position for the bar
    ax2.annotate(
        "",
        xy=(x2, y_pos),
        xytext=(x1, y_pos),
        arrowprops=dict(
            arrowstyle="|-|, widthA=0.3, widthB=0.3",
            color="black",
            lw=1,
        ),
        fontsize=10,
        color="grey",
        ha="center",
        va="center",
    )
    ax2.text(
        x1 * 0.5,
        y_pos * 1.3,
        "$\\beta$ fit range",
        fontsize=10,
        color="black",
        ha="left",
        va="bottom",
    )

    # Annotate lc_corr range
    x1, x2 = lc_corr_min, lc_corr_max  # your x-values
    y_pos = 2e-2  # y-position for the bar
    ax2.annotate(
        "",
        xy=(x2, y_pos),
        xytext=(x1, y_pos),
        arrowprops=dict(
            arrowstyle="|-|, widthA=0.3, widthB=0.3",
            color="black",
            lw=1,
        ),
        fontsize=10,
        color="grey",
        ha="center",
        va="center",
    )
    ax2.text(
        x1 * 1.6,
        y_pos * 1.3,
        "$\\lambda_C$ range",
        fontsize=10,
        color="black",
        ha="left",
        va="bottom",
    )

    # Create a manual legend
    handles = [
        plt.Line2D([0], [0], color="black", lw=2, label="Naive SF (entire interval)"),
        plt.Line2D(
            [0], [0], color="black", lw=1, alpha=0.25, label="Corrected SFs (subsets)"
        ),
    ]
    ax3.legend(
        handles=handles,
        loc="lower left",
        fontsize=8,
        frameon=True,
    )

    # Final layout adjustments
    plt.tight_layout()
    return fig


def _add_mission_info(ax, colors):
    """Add mission information text boxes."""
    # Voyager 1 info
    ax.text(
        0.03,
        0.97,
        "$\\bf{Voyager\ 1}$",
        transform=ax.transAxes,
        fontsize=10,
        color=colors["v1"],
        verticalalignment="top",
        # bbox=dict(
        #     boxstyle="round,pad=0.3",
        #     facecolor="white",
        #     edgecolor=colors["v1"],
        #     alpha=0.8,
        # ),
    )
    ax.text(
        0.04,
        0.88,
        "121-160au\n$⟨B⟩$ = 0.47 nT\n$⟨\delta b/B_0⟩$ = 0.22",
        transform=ax.transAxes,
        fontsize=9,
        color=colors["v1"],
        verticalalignment="top",
        # bbox=dict(
        #     boxstyle="round,pad=0.3",
        #     facecolor="white",
        #     edgecolor=colors["v1"],
        #     alpha=0.8,
        # ),
    )

    # Voyager 2 info
    ax.text(
        0.97,
        0.97,
        "$\\bf{Voyager\ 2}$",
        transform=ax.transAxes,
        fontsize=10,
        color=colors["v2"],
        verticalalignment="top",
        horizontalalignment="right",
        # bbox=dict(
        #     boxstyle="round,pad=0.3",
        #     facecolor="white",
        #     edgecolor=colors["v2"],
        #     alpha=0.8,
        # ),
    )
    ax.text(
        0.97,
        0.88,
        "119-135au\n$⟨B⟩$ = 0.57 nT\n$⟨\delta b/B_0⟩$ = 0.15nT",
        transform=ax.transAxes,
        fontsize=9,
        color=colors["v2"],
        verticalalignment="top",
        horizontalalignment="right",
        # bbox=dict(
        #     boxstyle="round,pad=0.3",
        #     facecolor="white",
        #     edgecolor=colors["v2"],
        #     alpha=0.8,
        # ),
    )


def _plot_corrected_sfs(ax, sfs_corrected, color, scale_factor=1):
    """Plot corrected structure functions."""
    for idx in sfs_corrected["int_index"].unique():
        subset = sfs_corrected[sfs_corrected["int_index"] == idx]
        ax.loglog(
            subset["lag_s"],
            subset["sf_2"] / scale_factor,
            alpha=0.25,
            color=color,
            linewidth=1,
        )


def _plot_corrected_es(ax, sfs_corrected, color, scale_factor=1):
    """Plot corrected equivalent spectra."""
    for idx in sfs_corrected["int_index"].unique():
        subset = sfs_corrected[sfs_corrected["int_index"] == idx]
        ax.loglog(
            subset["inverse_lag_s"],
            subset["sf_corrected_es"] * scale_factor,
            alpha=0.25,
            color=color,
            linewidth=1,
        )


def _add_k41_references_sf(ax, color):
    """Add Kolmogorov 2/3 power law references to structure function plot."""
    tau_range = np.logspace(np.log10(48), np.log10(2.5 * 365 * 86400), 100)

    # Two reference lines with different amplitudes
    ax.loglog(
        tau_range[:40],
        1e-5 * tau_range[:40] ** (2 / 3),
        color=color,
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
    )
    ax.loglog(
        tau_range,
        3.1e-8 * tau_range ** (2 / 3),
        color=color,
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
    )

    # Add K41 labels
    ax.text(2e2, 7e-4, "K41", rotation=30, alpha=0.7, fontsize=10, color=color)
    ax.text(1e7, 6e-4, "K41", rotation=30, alpha=0.7, fontsize=10, color=color)


def _add_k41_references_es(ax, freq_range, color):
    """Add Kolmogorov -5/3 power law references to equivalent spectra plot."""
    f_range = np.linspace(freq_range[-1], freq_range[0], 100)

    # Two reference lines with different amplitudes
    ax.loglog(
        f_range,
        3.1e-5 * f_range ** (-5 / 3),
        color=color,
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
    )
    ax.loglog(
        f_range,
        5e-10 * f_range ** (-5 / 3),
        color=color,
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
    )

    # Add K41 labels
    ax.text(1e-3, 7e-1, "K41", rotation=-40, alpha=0.7, fontsize=10, color=color)
    ax.text(1e-8, 1e2, "K41", rotation=-40, alpha=0.7, fontsize=10, color=color)


def _add_spatial_axis(ax, solar_wind_speed):
    """Add secondary x-axis for spatial scales."""
    ax_spatial = ax.twiny()
    lag_min, lag_max = ax.get_xlim()
    length_min, length_max = lag_min * solar_wind_speed, lag_max * solar_wind_speed
    ax_spatial.set_xlim(length_min, length_max)
    ax_spatial.set_xscale("log")
    ax_spatial.set_xlabel("$\ell$ (m)", fontsize=12)
    return ax_spatial


def _add_wavenumber_axis(ax, solar_wind_speed):
    """Add secondary x-axis for wavenumber."""
    ax_k = ax.twiny()
    freq_min, freq_max = ax.get_xlim()
    k_min, k_max = (
        2 * np.pi * freq_min / solar_wind_speed,
        2 * np.pi * freq_max / solar_wind_speed,
    )
    ax_k.set_xlim(k_min, k_max)
    ax_k.set_xscale("log")
    ax_k.set_xlabel("k (m$^{-1}$)", fontsize=12)
    return ax_k


def _add_scale_annotations(ax_time, ax_spatial):
    """Add time and length scale annotations."""
    AU = 1.496e11
    DI = 700000

    # Time scale annotations
    time_scales = [(48, "48s"), (365 * 86400, "1 year")]
    for time_val, label in time_scales:
        ax_time.annotate(
            "",
            xy=(time_val, 0),
            xytext=(time_val, 0.13),
            xycoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.3),
            annotation_clip=False,
        )
        ax_time.text(
            time_val,
            0.2,
            label,
            ha="center",
            va="top",
            fontsize=10,
            alpha=0.5,
            transform=ax_time.get_xaxis_transform(),
            clip_on=False,
        )

    # Length scale annotations
    length_scales = [(AU, "1 AU"), (DI, "$\\approx d_i$")]
    for length_val, label in length_scales:
        ax_spatial.annotate(
            "",
            xy=(length_val, 1),
            xytext=(length_val, 0.88),
            xycoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.3),
            annotation_clip=False,
        )
        ax_spatial.text(
            length_val,
            0.82,
            label,
            ha="center",
            va="bottom",
            transform=ax_spatial.get_xaxis_transform(),
            fontsize=10,
            alpha=0.5,
            clip_on=False,
        )


# Read the file voyager SF data exported from the notebook
# "../data/interim/voyager/vlism_sfs.pkl"

# Load the pickle file
vlism_sfs = read_pickle_file("data/interim/voyager/vlism_sfs.pkl")

# Unpack Voyager 1 data
lags_v1_hr = vlism_sfs["V1"]["lags"]
sf_v1_hr = vlism_sfs["V1"]["sf_hr"]
v1_sfs_corrected = vlism_sfs["V1"]["sfs_corrected"]
v1_res = vlism_sfs["V1"]["data_res"]
# v1_raw = vlism_sfs["V1"]["v1_raw"]

# Unpack Voyager 2 data
lags_v2_hr = vlism_sfs["V2"]["lags"]
sf_v2_hr = vlism_sfs["V2"]["sf_hr"]
v2_sfs_corrected = vlism_sfs["V2"]["sfs_corrected"]
v2_res = vlism_sfs["V2"]["data_res"]
# v2_raw = vlism_sfs["V1"]["v2_raw"]

# Convert to FULL SF to equivalent spectra
dt = 48  # seconds

es_v1_hr = sf_v1_hr * lags_v1_hr * dt / 6
es_v1_f = 1 / (lags_v1_hr * dt * 2)
es_v2_hr = sf_v2_hr * lags_v2_hr * dt / 6
es_v2_f = 1 / (lags_v2_hr * dt * 2)


# Get correlation lengths
# (Simply read dataframe of scalar stats)

# Read and process data
df_v1 = pd.read_csv("results/full/voyager1_corrected_metadata_NEW_RANGE.csv")

# Calculate time columns
df_v1["tce_s"] = df_v1["tce"] * df_v1["cadence"]
df_v1["ttu_s"] = df_v1["ttu"] * df_v1["cadence"]
df_v1["tce_days"] = df_v1["tce_s"] / (24 * 3600)
df_v1["ttu_hours"] = df_v1["ttu_s"] / 3600

# Read and process data
df_v2 = pd.read_csv("results/full/voyager2_corrected_metadata_NEW_RANGE.csv")

# Calculate time columns
df_v2["tce_s"] = df_v2["tce"] * df_v2["cadence"]
df_v2["ttu_s"] = df_v2["ttu"] * df_v2["cadence"]
df_v2["tce_days"] = df_v2["tce_s"] / (24 * 3600)
df_v2["ttu_hours"] = df_v2["ttu_s"] / 3600

df = pd.concat([df_v1, df_v2], ignore_index=True)
df_all_corr = df[df["gap_handling"] == "corrected_3d"]

lc_corr_min = df_all_corr.tce_s.min()
lc_corr_max = df_all_corr.tce_s.max()


# Get min and max of correlation lengths
# v1_min_corr_length = v1_scalars["corr_length"].min()
# v1_max_corr_length = v1_scalars["corr_length"].max()

# Add as annotations to the plot


# Example usage:
fig = create_voyager_analysis_plot(
    v1_res,
    v2_res,
    v1_sfs_corrected,
    v2_sfs_corrected,
    lags_v1_hr,
    sf_v1_hr,
    lags_v2_hr,
    sf_v2_hr,
    es_v1_f,
    es_v1_hr,
    es_v2_f,
    es_v2_hr,
    lc_corr_min,
    lc_corr_max,
)
plt.savefig("big_voyager_sfs_col.png", dpi=300, bbox_inches="tight")
# plt.show()


# Get fluctuation sizes
# Bx = v1_raw["BR"]
# By = v1_raw["BT"]
# Bz = v1_raw["BN"]

# Bx_mean = Bx.mean()
# By_mean = By.mean()
# Bz_mean = Bz.mean()

# # Calculate magnetic field magnitude B0
# B0 = np.sqrt(Bx_mean**2 + By_mean**2 + Bz_mean**2)
# print(f"B0: {B0:.3f} nT")
# # Calculate rms magnetic field fluctuations, db
# dbx = Bx - Bx_mean
# dby = By - By_mean
# dbz = Bz - Bz_mean
# db = np.sqrt(np.mean(dbx**2 + dby**2 + dbz**2))

# print(f"db: {db:.3f} nT")
# # Calculate the ratio of fluctuations to mean field
# ratio = db / B0
# print(f"Ratio of fluctuations to mean field: {ratio:.3f}")

# Bx = v2_raw["BR"]
# By = v2_raw["BT"]
# Bz = v2_raw["BN"]

# Bx_mean = Bx.mean()
# By_mean = By.mean()
# Bz_mean = Bz.mean()

# # Calculate magnetic field magnitude B0
# B0 = np.sqrt(Bx_mean**2 + By_mean**2 + Bz_mean**2)
# print(f"B0: {B0:.3f} nT")
# # Calculate rms magnetic field fluctuations, db
# dbx = Bx - Bx_mean
# dby = By - By_mean
# dbz = Bz - Bz_mean
# db = np.sqrt(np.mean(dbx**2 + dby**2 + dbz**2))

# print(f"db: {db:.3f} nT")
# # Calculate the ratio of fluctuations to mean field
# ratio = db / B0
# print(f"Ratio of fluctuations to mean field: {ratio:.3f}")
