import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.tsa.stattools as ts
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import fft, signal, stats

sys.path.append("external/Equivalent_Spectrum")
import external.Equivalent_Spectrum.equiv_spectrum as equiv_spectrum

# Set random seed for reproducibility
np.random.seed(42)

# TO-DO (see existing apps)
# - Don't compute un-needed periodogram, just get frequencies
# - Add Blackman-Tukey from Kea: should work with data gaps, and robust to noise?
# - Retain lines for original/underlying data (colour grey): save simulated data and corresponding stats, comment out, read in
# - Add periodic gaps (70 every 100 points)
# - Add sample size curves when missing data

# - Give to ChatGPT, ask to
#       - make interactive
#       - only re-compute when parameters change (i.e. don't compute original stats)
# - Publish online


def compute_es(data):
    f1 = data
    D = np.ndim(f1)
    grid_dims = np.shape(f1)
    N = np.min(grid_dims)
    L = len(f1)  # L = 2.*np.pi
    phys_dims = [L for _ in range(D)]
    dx = L / N
    dk = 2.0 * np.pi / L

    # Calculate the second order structure function
    lags, sf2, _ = compute_structure_function(data, lags=np.arange(1, L // 2))
    ell = lags * dx
    # Mark's code uses the following:
    # vell, sf2 = mpi_sf.mpi_sf(f1)
    # ell = vell[:,0]*dx

    # Bin and interpolate the second order structure function
    ell_b, sf2_b, _ = equiv_spectrum.bin_data(
        ell,
        sf2,
        bin_func=np.nanmean,
        cut_excess=True,
        nan_small=False,
        min_bin=dx,
        max_bin=L / 2.0,
        num_bins=32,
        bin_loc="true_center",
        log_space=True,
    )
    ell_b2 = ell_b[np.isfinite(sf2_b)]
    sf2_b2 = sf2_b[np.isfinite(sf2_b)]
    sf2_b = equiv_spectrum.log_log_interpolate(ell_b2, sf2_b2, ell_b)

    # Compare to FFT spectrum
    kvec, fek = equiv_spectrum.per_spectrum(f1, phys_dims)
    fek = fek * (dk / (2.0 * np.pi))
    ko, feko = equiv_spectrum.integrate_spectrum(kvec, fek, phys_dims)
    # Calculate the Uncorrected estimate, and the Debiased estimate
    ke, BfekS, fekS = equiv_spectrum.equiv_spectrum(ell_b, sf2_b, D, 1.0)

    return ko, feko, ke, BfekS, fekS, ell_b, sf2_b


def simulate_turbulence(n_points=86400, sampling_freq=1.0):
    """
    Simulate turbulence.

    Parameters:
    - n_points: Number of data points (default: 86400, representing 24 hours of 1 Hz data)

    Returns:
    - t: Time array in hours
    - y: Simulated time series data
    """

    # Time array (in hours)
    t = np.arange(n_points) * sampling_freq

    # 1. Generate turbulent component with approximate -5/3 power spectrum (Kolmogorov's law)
    # We'll use fractional Gaussian noise for this
    freqs = np.fft.rfftfreq(n_points, d=1.0 / sampling_freq)
    freqs[0] = freqs[1]  # Avoid division by zero

    # Generate complex amplitudes with power law spectrum
    amplitude = freqs ** (
        -5 / 6
    )  # We use -5/6 because we'll square the amplitude later
    phase = 2 * np.pi * np.random.random(len(freqs))

    # Create complex Fourier coefficients
    f_coeffs = amplitude * np.exp(1j * phase)
    f_coeffs[0] = 0  # Remove DC component

    # Generate the turbulent component via inverse FFT
    turbulence = np.fft.irfft(f_coeffs, n=n_points)

    # Scale the turbulence component
    turbulence = turbulence / np.std(turbulence) * 10

    return t, turbulence


def compute_structure_function(data, lags=None, max_lag=None):
    """
    Compute the second-order structure function of a time series.

    Parameters:
    - data: Input time series
    - lags: Array of lag values to calculate (if None, automatically determined)
    - max_lag: Maximum lag to calculate (default: 1/4 of data length)

    Returns:
    - lags: Lag array
    - sf2: Second-order structure function
    """
    n = len(data)

    if max_lag is None:
        max_lag = n // 2

    if lags is None:
        # Create logarithmically spaced lags for better visualization
        lags = np.unique(np.logspace(0, np.log10(max_lag), 100).astype(int))
        # lags = lags[lags > 0]  # Ensure no zero lag

    sf2 = np.zeros(len(lags))
    sf4 = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        # Calculate squared differences for all possible pairs at this lag
        diff = data[lag:] - data[:-lag]
        sf2[i] = np.nanmean(diff**2)
        sf4[i] = np.nanmean(diff**4)

    kurtosis = sf4 / (sf2**2)  # Excess kurtosis

    return lags, sf2, kurtosis


# Function to compute slopes to verify scaling laws
def compute_scaling_slope(x, y, range_start, range_end):
    """Compute the slope of log(y) vs log(x) in the specified range."""
    mask = (x >= range_start) & (x <= range_end)
    if np.sum(mask) < 2:
        return None

    logx = np.log(x)
    logy = np.log(y)

    # Linear regression
    A = np.vstack([logx, np.ones(len(logx))]).T
    slope, _ = np.linalg.lstsq(A, logy, rcond=None)[0]

    return slope


def make_tsa_plot(
    data="turbulence",
    seasonality_period=None,
    season_amplitude=10,
    n_points=10000,
    sampling_freq=1.0,
    linear_trend=True,
    linear_trend_amplitude=50,
    white_noise=False,
    white_noise_sigma=2,
    standardize=False,
    subtract_mean=False,
    remove_fraction_random=0.0,
    remove_fraction_periodic=0.1,  # Not yet implemented
    pwrl_min_freq=0.005,
    pwrl_max_freq=0.05,
):

    data = data  # Could be "white noise", "periodic", or "random walk"

    title = "TIME SERIES\n"

    # Generate the chosen data
    if data == "turbulence":
        t, data = simulate_turbulence(n_points, sampling_freq)
        title += "Simulated Turbulent Flow (-5/3 Power Law)"
    elif data == "fbm":
        data = np.load("external/Equivalent_Spectrum/example_data/example_fbm_1d.npy")
        t = np.arange(len(data)) / sampling_freq  # Create time array
        title += "Fractional Brownian Motion"
    elif data == "white noise":
        # Generate white noise data
        t = np.arange(n_points) / sampling_freq
        data = np.random.normal(0, white_noise_sigma, n_points)
        title += "White Noise"
    elif data == "periodic":
        # Generate periodic data
        t = np.arange(n_points) / sampling_freq
        frequency = 1 / seasonality_period if seasonality_period else 1
        data = season_amplitude * np.sin(2 * np.pi * frequency * t)
        title = f"Periodic Data (Period: {seasonality_period}, Amplitude: {season_amplitude})"
    elif data == "random walk":
        # Generate random walk data
        t = np.arange(n_points) / sampling_freq
        data = np.cumsum(np.random.normal(0, 1, n_points))
        title = "Random Walk"

    if seasonality_period:
        seasonality = season_amplitude * np.sin(2 * np.pi * t / seasonality_period)
        data += seasonality
        title += f"\n + Seasonality (Period: {seasonality_period}, Amplitude: {season_amplitude})"

    if linear_trend:
        # Add a linear trend
        linear_trend = np.linspace(0, 1, n_points) * linear_trend_amplitude
        data += linear_trend
        title += "\n + Linear Trend"

    if white_noise:
        # Generate white noise and calculate stats, add to data
        white_noise_data = np.random.normal(0, white_noise_sigma, n_points)

        _, psd_n = signal.periodogram(white_noise_data, fs=sampling_freq)
        psd_n = psd_n[1:] / (2 * np.pi)  # Exclude the zero frequency

        lags_n, sf2_n, _ = compute_structure_function(
            white_noise_data, max_lag=n_points // 2
        )

        acf_n = ts.acf(white_noise_data, nlags=n_points // 2)

        data += white_noise_data
        title += f"\n + White Noise ($\sigma_n$: {white_noise_sigma})"

    if standardize:
        # Normalize the data to have zero mean and unit variance
        data = (data - np.mean(data)) / np.std(data)
        title += "\n + Standardized"

    if subtract_mean:
        # Subtract the mean from the data
        data -= np.mean(data)
        title += "\n + Mean Subtracted"

    if remove_fraction_random > 0:
        # Remove random points from the data
        indices = np.random.choice(
            n_points, size=int(n_points * remove_fraction_random), replace=False
        )
        data[indices] = np.nan  # Set to NaN to simulate missing data
        title += f"\n + {remove_fraction_random*100:.1f}% Random Points Removed"

        # Create array, 0 if missing, 1 if not
        gap_signal = np.ones(n_points)
        gap_signal[indices] = 0  # Set to False where data is missing

        # Compute the power spectrum of the gaps
        _, psd_gaps = signal.periodogram(
            gap_signal, fs=sampling_freq, scaling="density"
        )
        psd_gaps = psd_gaps[1:] / (2 * np.pi)  # Exclude the zero frequency

        # Compute the autocovariance function of the gaps
        acf_gaps = ts.acf(gap_signal, nlags=n_points // 2, missing="conservative")

        lags_gaps, sf2_gaps, _ = compute_structure_function(
            gap_signal, max_lag=n_points // 2
        )

    data_var = np.nanvar(data)

    # Compute power spectrum with classical method - now just keeping freqs
    freqs, _ = signal.periodogram(data, fs=sampling_freq)

    freqs = 2 * np.pi * freqs  # Convert to angular frequency
    # Remove NaN values but keep time information
    mask = ~np.isnan(data)
    clean_data = data[mask]
    clean_t = t[mask]

    # Compute the Lomb-Scargle periodogram, allowing for unevenly spaced data
    psd = signal.lombscargle(clean_t, clean_data, freqs, normalize=False) / np.pi
    # this is equivalent to the following:
    # signal.periodogram(data, fs=sampling_freq) / (2 * np.pi)

    # Add the fitted slope line
    try:
        # Fit a power law to the power spectrum
        psd_fit = stats.linregress(
            np.log(freqs[(freqs > pwrl_min_freq) & (freqs < pwrl_max_freq)]),
            np.log(psd[(freqs > pwrl_min_freq) & (freqs < pwrl_max_freq)]),
        )

    # otherwise, raise the error and set to None
    except ValueError as e:
        print(f"Error fitting power law: {e}")
        psd_fit = None
        es_fit = None

    # Compute the power spectrum using the Fourier transform of ACF

    freqs = freqs[1:]  # Exclude the zero frequency
    psd = psd[1:]  # Exclude the zero frequency
    # Convert frequency to period (hours) for easier interpretation

    # Compute structure function
    lags, sf2, kurt = compute_structure_function(data, lags=np.arange(1, n_points // 2))

    # Add the fitted slope line
    try:
        # Fit a power law to the structure function
        sf_fit = stats.linregress(
            np.log(lags[(lags > 1 / pwrl_max_freq) & (lags < 1 / pwrl_min_freq)]),
            np.log(sf2[(lags > 1 / pwrl_max_freq) & (lags < 1 / pwrl_min_freq)]),
        )
    # otherwise, raise the error and set to None
    except ValueError as e:
        print(f"Error fitting power law: {e}")
        sf_fit = None

    # Compute the autocovariance function
    acf = ts.acf(data, nlags=n_points // 2, missing="conservative", adjusted=True)

    # Compute the acf from the structure function
    acf_from_sf = 1 - (sf2 / (2 * data_var))

    correlation_length = None
    # Compute the correlation length
    # correlation_length = np.argmax(
    #     acf < 1 / np.e
    # )  # Find the lag where ACF drops below 1/e

    # Compute the power spectrum from the structure function
    psd_k, psd_k_new, es_k, es_biased, es, lags_binned, sf2_binned = compute_es(data)

    # Create a nice figure with both analyses

    palette = {"psd": "#d95f02", "sf": "#1b9e77", "acf": "#7570b3"}

    # Set a consistent style
    sns.set_style("ticks")
    plt.rc("font", family="Arial")
    # plt.rc("axes", labelsize=12)
    # plt.rc("xtick", labelsize=10)
    # plt.rc("ytick", labelsize=10)
    # plt.rc("legend", fontsize=10)

    ############################################

    fig = plt.figure(figsize=(13, 6))

    # Top panel spans entire first row (row 0, columns 0-2)
    ax_top = plt.subplot2grid((2, 4), (0, 0), colspan=4)

    # Bottom row panels
    ax_bottom = [
        plt.subplot2grid((2, 4), (1, 0)),  # row 1, col 0
        plt.subplot2grid((2, 4), (1, 1)),  # row 1, col 1
        plt.subplot2grid((2, 4), (1, 2)),  # row 1, col 2
        plt.subplot2grid((2, 4), (1, 3)),  # row 1, col 2
    ]

    # Plot the time series
    ax_top.plot(t, data, linewidth=0.8, color="black")
    ax_top.set_xlabel("Time (s)")
    ax_top.set_ylabel("")

    ax_top.set_title(title, fontweight="bold")

    # Plot the structure function
    # 1. Log-log plot showing scaling regions
    ax_bottom[0].axhline(
        y=2 * data_var, label="$2\sigma^2$", alpha=0.5, lw=0.3, color="black"
    )
    ax_bottom[0].loglog(lags, sf2, linewidth=2, color=palette["sf"], alpha=0.3)
    ax_bottom[0].loglog(
        lags_binned,
        sf2_binned,
        linewidth=0.8,
        color=palette["sf"],
        marker="o",
        markersize=2,
    )

    if sf_fit is not None:

        slope = sf_fit.slope
        intercept = sf_fit.intercept
        x_fit = np.linspace(1 / pwrl_min_freq, 1 / pwrl_max_freq, 100)
        y_fit = np.exp(slope * np.log(x_fit) + intercept)
        ax_bottom[0].loglog(
            x_fit,
            y_fit * 2,
            color=palette["sf"],
            label=f"Fitted Slope: {slope:.2f}",
            lw=2.5,
        )

    ax_bottom[0].set_xlabel("Lag (s)")
    ax_bottom[0].set_ylabel("$S_2$")
    ax_bottom[0].set_title(
        "STRUCTURE FUNCTION (binned)", fontweight="bold", color=palette["sf"]
    )
    #
    # Plot the autocovariance function
    ax_bottom[1].axhline(y=0, color="black", alpha=0.3, lw=0.5)
    ax_bottom[1].plot(acf, linewidth=2, color=palette["acf"], alpha=1)
    ax_bottom[1].plot(
        acf_from_sf,
        color=palette["sf"],
        label="ACF from SF",
        linewidth=0.8,
    )
    ax_bottom[1].set_xlabel("Lag (s)")
    ax_bottom[1].set_ylabel("$R$")
    ax_bottom[1].set_title(
        "AUTOCORRELATION FUNCTION", fontweight="bold", color=palette["acf"]
    )

    inset_xmin = 0
    inset_xmax = 25
    inset_ymin = acf[inset_xmax]
    inset_ymax = 1
    axins = inset_axes(ax_bottom[1], width="30%", height="30%", loc="upper right")
    axins.plot(
        acf, marker="o", color=palette["acf"], alpha=1, markersize=1, linewidth=0.5
    )
    axins.set_xlim(inset_xmin, inset_xmax)
    axins.set_ylim(inset_ymin, inset_ymax)
    # Reduce font size
    axins.tick_params(labelsize=8)

    # Plot the power spectrum (Lomb-Scargle periodogram)
    ax_bottom[2].loglog(
        freqs / (2 * np.pi),  # Convert frequency to Hz
        psd,
        color=palette["psd"],
        alpha=0.8,  # freqs * (2 * np.pi), psd / (2 * np.pi)
    )
    # ax_bottom[2].loglog(
    #     psd_k / (2 * np.pi),  # Convert frequency to Hz,
    #     psd_k_new,
    #     color="blue",
    #     label="Mark's PSD",
    #     alpha=0.5,
    # )
    ax_bottom[2].loglog(
        es_k / (2 * np.pi),
        es,
        marker="o",
        markersize=2,
        color=palette["sf"],
        alpha=0.8,
        lw=1,
        label="Equivalent Spectrum",
        # linewidth=0.8,
    )

    ax_bottom[3].plot(lags, kurt, color="#e7298a", alpha=0.8, linewidth=1.5)
    ax_bottom[3].axhline(
        y=3, color="black", linestyle="--", label="Normal Kurtosis", alpha=0.5
    )
    ax_bottom[3].set_xlabel("Lag (s)")
    ax_bottom[3].set_ylabel("$S_4/S_2^2$")
    ax_bottom[3].set_title("KURTOSIS", fontweight="bold", color="#e7298a")

    if psd_fit is not None:

        slope = psd_fit.slope
        intercept = psd_fit.intercept

        x_fit = np.linspace(pwrl_min_freq, pwrl_max_freq, 100)
        y_fit = np.exp(slope * np.log(x_fit) + intercept)
        ax_bottom[2].loglog(
            x_fit,
            y_fit * 5,
            color=palette["psd"],
            label=f"Fitted Slope: {slope:.2f}",
            lw=2.5,
        )

    ax_bottom[2].set_xlabel("Frequency (Hz)")
    ax_bottom[2].set_ylabel("$E$")
    ax_bottom[2].set_title("POWER SPECTRUM", fontweight="bold", color=palette["psd"])

    if seasonality_period is not None:
        # Add vertical lines at the seasonality period/frequency
        ax_bottom[2].axvline(
            x=1 / seasonality_period,
            color="gray",
            label="Seasonality Frequency",
            alpha=0.5,
        )

        ax_bottom[0].axvline(
            x=seasonality_period, color="gray", label="Seasonality Period", alpha=0.5
        )
        ax_bottom[1].axvline(
            x=seasonality_period, color="gray", label="Seasonality Period", alpha=0.5
        )

    if correlation_length:
        ax_bottom[1].axvline(
            x=correlation_length,
            color="red",
            linestyle="--",
            label=f"$\lambda_C$: {correlation_length}s",
        )

    if white_noise:
        ax_bottom[2].loglog(
            freqs / (2 * np.pi),
            psd_n,
            linewidth=1.5,
            color="gray",
            alpha=0.5,
            label="Noise Spectrum",
        )
        ax_bottom[1].plot(
            acf_n,
            linewidth=1.5,
            color="gray",
            alpha=0.5,
            label="Noise ACF$\\approx 0$",
        )
        ax_bottom[0].loglog(
            lags_n,
            sf2_n,
            linewidth=1.5,
            color="gray",
            alpha=0.5,
            label="Noise SF$\\approx 2\sigma^2_n$",
        )

    if remove_fraction_random > 0:
        # Plot the power spectrum of the gaps
        # 1. Log-log plot showing the power law behavior
        ax_bottom[2].loglog(
            freqs / (2 * np.pi),
            psd_gaps,
            linewidth=1,
            color="purple",
            alpha=0.2,
            label="Gap Spectrum",
        )

        ax_bottom[1].plot(
            acf_gaps,
            linewidth=1.5,
            color="purple",
            alpha=0.2,
            label="Gap ACF",
        )
        ax_bottom[0].loglog(
            lags_gaps,
            sf2_gaps,
            linewidth=1.5,
            color="purple",
            alpha=0.2,
            label="Gap SF",
        )

    ax_bottom[0].legend()
    ax_bottom[1].legend(loc="center right")
    ax_bottom[2].legend()

    plt.tight_layout()


if __name__ == "__main__":
    make_tsa_plot(
        # Simulation parameters
        data="turbulence",  # Options: "turbulence", "white noise", "periodic", "random walk"
        seasonality_period=None,  # Set None for no seasonality
        season_amplitude=1,
        n_points=5000,
        sampling_freq=1.0,  # Hz
        linear_trend=False,
        linear_trend_amplitude=50,
        white_noise=False,
        white_noise_sigma=0.3,
        standardize=False,
        subtract_mean=False,
        remove_fraction_random=0,
        remove_fraction_periodic=0.1,  # Not yet implemented
    )
    plt.show()
    # plt.savefig("cool_figs/fbm_w_periodic_1000.png", dpi=300, bbox_inches="tight")


# NOTES:
# SF does not line up with PSD exactly due to inverse procedure
# You will get some extra points at large scales because SF gives you k=0,
# which is not necessarily well-defined for some time series
