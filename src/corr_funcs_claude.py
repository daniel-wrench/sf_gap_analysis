from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit


class TimeUnit(Enum):
    """Enumeration of supported time units for display."""

    SECONDS = auto()
    MINUTES = auto()
    HOURS = auto()
    DAYS = auto()


# Unit conversion factors (relative to seconds)
TIME_UNIT_FACTORS = {
    TimeUnit.SECONDS: 1,
    TimeUnit.MINUTES: 60,
    TimeUnit.HOURS: 3600,
    TimeUnit.DAYS: 86400,
}

# Unit labels for plotting
TIME_UNIT_LABELS = {
    TimeUnit.SECONDS: "s",
    TimeUnit.MINUTES: "min",
    TimeUnit.HOURS: "hr",
    TimeUnit.DAYS: "days",
}


def compute_nd_acf(
    time_series: List[pd.Series],
    nlags: Optional[int] = None,
    plot: bool = False,
    time_unit: TimeUnit = TimeUnit.SECONDS,
    distance_axis: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the autocorrelation function for a scalar or vector time series.

    Args:
        time_series: List of 1 (scalar) or 3 (vector) pd.Series. The function automatically detects
                    the cadence if timestamped index, otherwise dt = 1s.
        nlags: The number of lags to calculate the ACF up to. If None, compute the full ACF.
        plot: Whether to plot the ACF.
        time_unit: The time unit to use for plotting and returned time lags.

    Returns:
        Tuple of:
            - time_lags: The x-values of the ACF, in the specified time unit
            - acf: The values of the ACF from lag 0 to nlags
    """
    # Convert the time series into a numpy array
    np_array = np.array(time_series)

    # Determine the number of lags if not specified
    if nlags is None:
        nlags = len(np_array[0]) - 1

    # Calculate the ACF based on the dimensionality of the data
    if np_array.shape[0] == 3:
        # Vector data: average ACF across the 3 components
        acf = (
            sum(
                sm.tsa.acf(np_array[i], fft=True, nlags=nlags, missing="conservative")
                for i in range(3)
            )
            / 3
        )
    elif np_array.shape[0] == 1:
        # Scalar data
        acf = sm.tsa.acf(np_array[0], fft=True, nlags=nlags, missing="conservative")
    else:
        raise ValueError(
            "Array is not 3D or 1D. If after a 1D acf, try putting square brackets around the pandas series in np.array()"
        )

    # Determine the time step (dt) from the data
    if isinstance(time_series[0].index, pd.DatetimeIndex):
        # Get the cadence of the data
        dt = (time_series[0].index[1] - time_series[0].index[0]).total_seconds()
    else:
        # If not, assume 1 second cadence
        dt = 1

    # Calculate time lags in the requested unit
    unit_factor = TIME_UNIT_FACTORS[time_unit]
    time_lags = np.arange(0, nlags + 1) * dt / unit_factor

    # Optional plotting
    if plot:
        fig, ax = plt.subplots(constrained_layout=True)

        unit_label = TIME_UNIT_LABELS[time_unit]
        ax.plot(time_lags, acf)
        ax.set_xlabel(f"$\\tau$ ({unit_label})")
        ax.set_ylabel("Autocorrelation")

        # For plotting secondary axes
        def time_to_lag(x):
            return x * unit_factor / dt

        def lag_to_time(x):
            return x * dt / unit_factor

        secax_x = ax.secondary_xaxis("top", functions=(time_to_lag, lag_to_time))
        secax_x.set_xlabel("$\\tau$ (lag)")

        if distance_axis:
            # Assuming 400 km/s is the conversion factor
            def time_to_km(x):
                return x * unit_factor * 400  # Convert to seconds first, then to km

            def km_to_time(x):
                return (
                    x / 400 / unit_factor
                )  # Convert to seconds first, then to the selected unit

            secax_x2 = ax.secondary_xaxis(-0.2, functions=(time_to_km, km_to_time))
            secax_x2.set_xlabel("$r$ (km)")

        plt.show()

    return time_lags, acf


def compute_outer_scale_exp_trick(
    autocorrelation_x: np.ndarray,
    autocorrelation_y: np.ndarray,
    plot: bool = False,
    time_unit: TimeUnit = TimeUnit.SECONDS,
    distance_axis: bool = False,
) -> Union[float, Tuple[float, plt.Figure, plt.Axes]]:
    """
    Compute the correlation scale through the "1/e" estimation method.

    Args:
        autocorrelation_x: X values (time lags) of the autocorrelation function.
        autocorrelation_y: Y values of the autocorrelation function.
        plot: Whether to plot the tce.
        time_unit: The time unit to use for plotting and returned value.

    Returns:
        If plot=False: correlation scale in specified time unit.
        If plot=True: tuple of (correlation scale, figure, axes).
    """
    # unit_factor = TIME_UNIT_FACTORS[time_unit]
    unit_label = TIME_UNIT_LABELS[time_unit]

    # Convert times from raw seconds to the specified unit for display
    display_factor = 1  # Determines if we show values as 10^3 unit or direct values
    if np.max(autocorrelation_x) > 1000 and time_unit == TimeUnit.SECONDS:
        display_factor = 1000
        display_unit = f"$10^3$ {unit_label}"
    else:
        display_unit = unit_label

    # Find the correlation scale (where ACF = 1/e)
    x_opt = None
    for i, j in zip(autocorrelation_y, autocorrelation_x):
        if i <= np.exp(-1):
            # Linear interpolation to find more precise intercept
            idx_2 = np.where(autocorrelation_x == j)[0]
            idx_1 = idx_2 - 1
            x2 = autocorrelation_x[idx_2]
            x1 = autocorrelation_x[idx_1]
            y1 = autocorrelation_y[idx_1]
            y2 = autocorrelation_y[idx_2]
            x_opt = x1 + ((y1 - np.exp(-1)) / (y1 - y2)) * (x2 - x1)
            break

    if x_opt is None:
        return np.nan

    # Round the result to 3 decimal places
    tce = round(x_opt[0], 3)

    # Optional plotting
    if plot:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(5, 2.5), constrained_layout=True)

            # Plot ACF
            ax.plot(
                autocorrelation_x / display_factor,
                autocorrelation_y,
                c="black",
                label="Autocorrelation",
                lw=0.5,
            )
            ax.set_xlabel(f"$\\tau$ ({display_unit})")
            ax.set_ylabel("$R(\\tau)$")

            if distance_axis:
                # Secondary x-axis for distance
                def time_to_km(x):
                    return x * display_factor * 400 / 1e6  # Convert to 10^6 km

                def km_to_time(x):
                    return x / display_factor / 400 * 1e6

                secax_x2 = ax.secondary_xaxis("top", functions=(time_to_km, km_to_time))
                secax_x2.set_xlabel("$r$ ($10^6$ km)")
                secax_x2.tick_params(which="both", direction="in")

            # Plot the 1/e line and correlation scale
            formatted_value = f"{tce:.1f}" if tce < 10 else f"{tce:.0f}"
            ax.axhline(
                np.exp(-1),
                color="black",
                ls="--",
                label=f"$1/e\\rightarrow\\lambda_C^{{1/e}}$={formatted_value} {unit_label}",
            )
            ax.axvline(tce / display_factor, color="black", ls="--")
            ax.tick_params(which="both", direction="in")

            return tce, fig, ax
        except Exception as e:
            print(f"Error in plotting: {e}")
            return tce
    else:
        return tce


def exp_fit(r, lambda_c):
    """
    Fit function for determining correlation scale, through the optimal lambda_c value.

    Args:
        r: The time lag.
        lambda_c: The correlation scale parameter.

    Returns:
        The exponential fit function value.
    """
    return np.exp(-1 * r / lambda_c)


def para_fit(x, a):
    """
    Fit function for determining Taylor scale, through the optimal lambda_c value.

    Args:
        x: The time lag.
        a: The fit parameter.

    Returns:
        The parabolic fit function value.
    """
    return a * x**2 + 1


def compute_outer_scale_exp_fit(
    time_lags: np.ndarray,
    acf: np.ndarray,
    time_to_fit: float,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    plot: bool = False,
    time_unit: TimeUnit = TimeUnit.SECONDS,
) -> Union[float, Tuple[float, plt.Figure, plt.Axes]]:
    """
    Compute the correlation scale by fitting an exponential function.

    Args:
        time_lags: X values (time lags) of the autocorrelation function in seconds.
        acf: Y values of the autocorrelation function.
        seconds_to_fit: How many seconds of data to use for the fit.
        fig: Existing figure to plot on (optional).
        ax: Existing axes to plot on (optional).
        plot: Whether to plot the result.
        initial_guess: Initial guess for the correlation scale in seconds.
        time_unit: The time unit to use for display and returned value.

    Returns:
        If plot=False: correlation scale in the specified time unit.
        If plot=True: tuple of (correlation scale, figure, axes).
    """
    dt = time_lags[1] - time_lags[0]
    num_lags_for_lambda_c_fit = int(time_to_fit / dt)

    # Initial guess for the correlation scale is given by half time to fit = 1/e estimate
    initial_guess = time_to_fit / 2

    # Curve fitting to find lambda_c
    c_opt, _ = curve_fit(
        exp_fit,
        time_lags[:num_lags_for_lambda_c_fit],
        acf[:num_lags_for_lambda_c_fit],
        p0=initial_guess,
    )
    lambda_c = c_opt[0]

    # Convert to the requested time unit
    unit_factor = TIME_UNIT_FACTORS[time_unit]
    unit_label = TIME_UNIT_LABELS[time_unit]
    lambda_c_in_unit = lambda_c / unit_factor

    # Optional plotting
    if plot:
        # Use existing figure/axis if provided, else create new ones
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 2.5), constrained_layout=True)

        # Determine the display factor for x-axis
        display_factor = (
            1000 if np.max(time_lags) > 1000 and time_unit == TimeUnit.SECONDS else 1
        )

        # Plot the exponential fit
        formatted_value = (
            f"{lambda_c_in_unit:.1f}"
            if lambda_c_in_unit < 10
            else f"{lambda_c_in_unit:.0f}"
        )
        fit_times = np.array(range(int(time_to_fit)))
        ax.plot(
            fit_times / display_factor,
            exp_fit(fit_times, *c_opt),
            label=f"Exp. fit$\\rightarrow\\lambda_C^{{\\mathrm{{fit}}}}$={formatted_value} {unit_label}",
            lw=3,
            c="black",
        )

        return lambda_c_in_unit, fig, ax
    else:
        return lambda_c_in_unit


def compute_outer_scale_integral(
    time_lags: np.ndarray,
    acf: np.ndarray,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    plot: bool = False,
    time_unit: TimeUnit = TimeUnit.SECONDS,
) -> Union[Optional[float], Tuple[float, plt.Figure, plt.Axes]]:
    """
    Compute the correlation scale using the integral method.

    Args:
        time_lags: X values (time lags) of the autocorrelation function in seconds.
        acf: Y values of the autocorrelation function.
        fig: Existing figure to plot on (optional).
        ax: Existing axes to plot on (optional).
        plot: Whether to plot the result.
        time_unit: The time unit to use for display and returned value.

    Returns:
        If plot=False: correlation scale in the specified time unit, or None if no zero crossing is found.
        If plot=True: tuple of (correlation scale, figure, axes).
    """
    dt = time_lags[1] - time_lags[0]

    # Find where ACF changes sign (crosses zero)
    sign_changes = np.where(np.diff(np.signbit(acf)))[0]

    if len(sign_changes) == 0:
        return None  # No zero crossing found

    # Get index just before first zero crossing
    idx_before = sign_changes[0]

    # Computing integral up to that index
    tci = np.sum(acf[:idx_before]) * dt

    # Convert to the requested time unit
    unit_label = TIME_UNIT_LABELS[time_unit]

    # Optional plotting
    if plot:
        # Use existing figure/axis if provided, else create new ones
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 2.5), constrained_layout=True)

        # Determine the display factor for x-axis
        display_factor = (
            1000 if np.max(time_lags) > 1000 and time_unit == TimeUnit.SECONDS else 1
        )
        display_unit = f"$10^3$ {unit_label}" if display_factor == 1000 else unit_label

        # Plot the integral region
        formatted_value = f"{tci:.1f}" if tci < 10 else f"{tci:.0f}"
        ax.fill_between(
            time_lags / display_factor,
            0,
            acf,
            where=(acf > 0) & (time_lags < time_lags[idx_before]),
            color="black",
            alpha=0.2,
            label=f"Integral$\\rightarrow\\lambda_C^{{\\mathrm{{int}}}}$={formatted_value} {unit_label}",
        )
        ax.set_xlabel(f"$\\tau$ ({display_unit})")
        ax.tick_params(which="both", direction="in")

        # Plot the legend
        ax.legend(loc="upper right")

        return tci, fig, ax
    else:
        return tci


def compute_all_correlation_scales(
    time_series: List[pd.Series],
    seconds_to_fit: float = 10000,
    nlags: Optional[int] = None,
    time_unit: TimeUnit = TimeUnit.SECONDS,
    xmax: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute and plot all three correlation scale methods in a single figure.

    Args:
        time_series: List of 1 (scalar) or 3 (vector) pd.Series.
        nlags: The number of lags to calculate the ACF up to.
        seconds_to_fit: How many seconds of data to use for the exponential fit.
        initial_guess: Initial guess for the correlation scale in seconds.
        time_unit: The time unit to use for display and returned values.

    Returns:
        Dictionary of results including the three correlation scales and the figure.
    """

    # Determine the number of lags if not specified
    if nlags is None:
        nlags = len(time_series[0]) - 1

    # Compute the ACF
    time_lags, acf = compute_nd_acf(time_series, nlags, plot=False, time_unit=time_unit)

    # Start with the 1/e method and create figure
    corr_scale_exp_trick, fig, ax = compute_outer_scale_exp_trick(
        time_lags, acf, plot=True, time_unit=time_unit
    )

    # Add exponential fit method
    corr_scale_exp_fit, fig, ax = compute_outer_scale_exp_fit(
        time_lags,
        acf,
        corr_scale_exp_trick * 2,
        fig,
        ax,
        plot=True,
        time_unit=time_unit,
    )

    # Add integral method
    corr_scale_integral, fig, ax = compute_outer_scale_integral(
        time_lags, acf, fig, ax, plot=True, time_unit=time_unit
    )

    # Finalize the plot
    ax.legend(loc="upper right")
    if xmax is not None:
        ax.set_xlim(0, xmax)
    plt.tight_layout()

    # Return results
    return {
        "correlation_scale_1/e": corr_scale_exp_trick,
        "correlation_scale_exp_fit": corr_scale_exp_fit,
        "correlation_scale_integral": corr_scale_integral,
        "figure": fig,
    }
