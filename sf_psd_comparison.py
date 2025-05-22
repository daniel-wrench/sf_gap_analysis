import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.tsa.stattools as ts
from scipy import signal, stats

# Set random seed for reproducibility
np.random.seed(42)

# TO-DO (see existing apps)

# - gaps (random or periodic, in which case switch to Lomb-Scargle for PSD)
# (in each case, retain original/underlying data for comparison)


def simulate_turbulence(
    n_points=86400, sampling_freq=1.0, seasonality=None, season_amplitude=10
):
    """
    Simulate a time series representing turbulent atmospheric data with a daily seasonality.

    Parameters:
    - n_points: Number of data points (default: 86400, representing 24 hours of 1 Hz data)
    - sampling_freq: Sampling frequency in Hz (default: 1 Hz)

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

    # Add a linear trend
    linear_trend = np.linspace(0, 1, n_points)
    turbulence += linear_trend

    if seasonality is None:
        return t, turbulence

    else:
        # Add a periodic component
        seasonality = season_amplitude * np.sin(2 * np.pi * t / seasonality_period)

        return t, turbulence + seasonality


def compute_power_spectrum(data, sampling_freq=1.0):
    """
    Compute the power spectral density of a time series.

    Parameters:
    - data: Input time series
    - sampling_freq: Sampling frequency in Hz

    Returns:
    - freqs: Frequency array
    - psd: Power spectral density
    """
    # Use Welch's method to estimate PSD
    freqs, psd = signal.welch(data, fs=sampling_freq)
    return freqs, psd


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
        max_lag = n // 4

    if lags is None:
        # Create logarithmically spaced lags for better visualization
        lags = np.unique(np.logspace(0, np.log10(max_lag), 100).astype(int))
        lags = lags[lags > 0]  # Ensure no zero lag

    sf2 = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        # Calculate squared differences for all possible pairs at this lag
        diff = data[lag:] - data[:-lag]
        sf2[i] = np.mean(diff**2)

    return lags, sf2


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


# Simulation parameters
seasonality_period = None
season_amplitude = 10
n_points = 10000  # 24 hours of data at 1 Hz
sampling_freq = 1.0  # Hz
linear_trend = False
linear_trend_amplitude = 100
white_noise = True
white_noise_std = 0.8

# Generate the simulated data
t, data = simulate_turbulence(
    n_points, sampling_freq, seasonality_period, season_amplitude
)

if linear_trend:
    # Add a linear trend
    data += np.linspace(0, 1, n_points) * linear_trend_amplitude

if white_noise:
    # Add white noise
    data += np.random.normal(0, white_noise_std, n_points)

# Compute power spectrum
freqs, psd = signal.periodogram(data, fs=sampling_freq)
freqs = freqs[1:]  # Exclude the zero frequency
psd = psd[1:]  # Exclude the zero frequency
# Convert frequency to period (hours) for easier interpretation


# Compute the noise stats, if applicable
if white_noise:
    # Generate white noise
    white_noise_data = np.random.normal(0, white_noise_std, n_points)

    freqs_n, psd_n = signal.periodogram(white_noise_data, fs=sampling_freq)
    freqs_n = freqs_n[1:]  # Exclude the zero frequency
    psd_n = psd_n[1:]  # Exclude the zero frequency

    lags_n, sf2_n = compute_structure_function(white_noise_data, max_lag=n_points)

    acf_n = ts.acf(white_noise_data, nlags=n_points)

# Compute structure function
lags, sf2 = compute_structure_function(data, max_lag=n_points)

# Compute the autocovariance function
acf = ts.acf(data, nlags=n_points)

correlation_length = None
# Compute the correlation length
# correlation_length = np.argmax(
#     acf < 1 / np.e
# )  # Find the lag where ACF drops below 1/e


pwrl_min_freq = 0.01
pwrl_max_freq = 0.1


# Create a nice figure with both analyses

fig, ax = plt.subplots(2, 2, figsize=(8, 6))

ax = ax.flatten()

# Set a consistent style
sns.set_style("ticks")
plt.rc("font", family="Arial")
# plt.rc("axes", labelsize=12)
# plt.rc("xtick", labelsize=10)
# plt.rc("ytick", labelsize=10)
# plt.rc("legend", fontsize=10)

# Plot the time series
ax[0].plot(t, data, linewidth=1, color="darkblue", alpha=0.8)
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("")

title = "Simulated Turbulent Flow"

if seasonality_period:
    title += f"\n + Seasonality (Period: {seasonality_period}, Amplitude: {season_amplitude})"

if linear_trend:
    title += "\n + Linear Trend"

if white_noise:
    title += f"\n + White Noise ($\sigma^2_n$: {white_noise_std})"

ax[0].set_title(title, fontweight="bold")

# Plot the power spectrum
# 1. Log-log plot showing the power law behavior
ax[1].loglog(freqs, psd, linewidth=1.5, color="darkred", alpha=0.8)


# Add the fitted slope line
slope = stats.linregress(
    np.log(freqs)[(freqs > pwrl_min_freq) & (freqs < pwrl_max_freq)],
    np.log(psd)[(freqs > pwrl_min_freq) & (freqs < pwrl_max_freq)],
).slope

if slope is not None:
    x_fit = np.linspace(pwrl_min_freq, pwrl_max_freq, 100)
    y_fit = np.exp(np.log(freqs[-1]) + slope * np.log(x_fit / freqs[-1]))
    ax[1].loglog(
        x_fit, y_fit, color="purple", linestyle="--", label=f"Fitted Slope: {slope:.2f}"
    )


ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("$E$")
ax[1].set_title("Power Spectrum", fontweight="bold")

# Plot the structure function
# 1. Log-log plot showing scaling regions
ax[3].loglog(lags, sf2, linewidth=1.5, color="darkgreen", alpha=0.8)

# Add the fitted slope line
slope = stats.linregress(
    np.log(lags)[(lags > 1 / pwrl_max_freq) & (lags < 1 / pwrl_min_freq)],
    np.log(sf2)[(lags > 1 / pwrl_max_freq) & (lags < 1 / pwrl_min_freq)],
).slope

if slope is not None:
    x_fit = np.linspace(1 / pwrl_min_freq, 1 / pwrl_max_freq, 100)
    y_fit = np.exp(np.log(sf2[0]) + slope * np.log(x_fit / lags[0]))
    ax[3].loglog(
        x_fit, y_fit, color="purple", linestyle="--", label=f"Fitted Slope: {slope:.2f}"
    )

ax[3].set_xlabel("Lag (s)")
ax[3].set_ylabel("$S_2$")
ax[3].set_title("Structure Function", fontweight="bold")
ax[3].axhline(y=2 * np.var(data), color="purple", linestyle=":", label="$2\sigma^2$")

# Plot the autocovariance function
ax[2].plot(acf, linewidth=1.5, color="darkorange", alpha=0.8)
ax[2].set_xlabel("Lag (s)")
ax[2].set_ylabel("$R$")
ax[2].set_title("Autocorrelation Function", fontweight="bold")
# ax[2].axhline(0, alpha=0.2, c="black")
# ax[2].axhline(y=np.var(data), color="purple", linestyle="--", label="Variance")

if seasonality_period is not None:
    # Add vertical lines at the seasonality period/frequency
    ax[1].axvline(
        x=1 / seasonality_period,
        color="gray",
        label="Seasonality Frequency",
    )
    ax[3].axvline(x=seasonality_period, color="gray", label="Seasonality Period")
    ax[2].axvline(x=seasonality_period, color="gray", label="Seasonality Period")


if correlation_length:
    ax[2].axvline(
        x=correlation_length,
        color="red",
        linestyle="--",
        label=f"$\lambda_C$: {correlation_length}s",
    )

if white_noise:
    ax[1].loglog(
        freqs_n, psd_n, linewidth=1.5, color="gray", alpha=0.5, label="Noise Spectrum"
    )
    ax[2].plot(
        acf_n,
        linewidth=1.5,
        color="gray",
        alpha=0.5,
        label="Noise ACF$\\approx 0$",
    )
    ax[3].loglog(
        lags_n,
        sf2_n,
        linewidth=1.5,
        color="gray",
        alpha=0.5,
        label="Noise SF$\\approx 2\sigma^2_n$",
    )

ax[1].legend()
ax[2].legend()
ax[3].legend()

plt.tight_layout()
# plt.suptitle(
#     "Comparison of Power Spectrum and Structure Function Analysis",
#     fontsize=16,
#     fontweight="bold",
#     y=1.02,
# )
plt.show()
