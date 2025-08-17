# Add .. to the path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fbm import FBM
from scipy import signal, stats
from scipy.interpolate import interp1d
from scipy.stats import linregress

sys.path.append("..")
import src.sf_funcs as sf_funcs
import src.utils as utils

plt.rc("font", family="Arial")


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
    # If any nan values, fill with 1 and print a warning
    if np.isnan(full_y_binned).any():
        print("Warning: NaN values found in smoothed correction. Filling with 1.")
        full_y_binned = np.nan_to_num(full_y_binned, nan=1)

    interp_func = interp1d(
        full_bins, full_y_binned, kind="cubic", fill_value="extrapolate"
    )

    smoothed_interp = interp_func(x)

    return smoothed_interp


def compute_scaling(sf_df, correction_lookup, n_bins=25):
    """Compute the scaling factor for the structure function based on a lookup table.
    This function applies the correction factor to the original data based on the
    provided lookup table, which contains the scaling factors for different bins.

    Parameters:

    - inputs: DataFrame containing the structure function data
    - dim: Dimension of the lookup table (2D or 3D)
    - correction_lookup: Dictionary containing the lookup table with scaling factors
    - n_bins: Number of bins in the lookup table (default is 25)

    Returns:

    - inputs: DataFrame with the scaling factor columns added

    """

    # Validate correction_lookup structure
    required_keys = [
        "xedges",
        "yedges",
        "zedges",
        "scaling",
        "scaling_lower",
        "scaling_upper",
    ]

    for key in required_keys:
        if key not in correction_lookup:
            raise ValueError(f"Missing required key: {key}")

    # Extract bins and scaling factors from the lookup table
    xedges = correction_lookup["xedges"]
    yedges = correction_lookup["yedges"]
    zedges = correction_lookup["zedges"]
    scaling = correction_lookup["scaling"]
    scaling_lower = correction_lookup["scaling_lower"]
    scaling_upper = correction_lookup["scaling_upper"]

    # Compute bin indices for the interpolated SF
    x = sf_df["lag"].values
    y = sf_df["gp"].values
    z = sf_df["sf"].values
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, n_bins - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, n_bins - 1)
    zidx = np.clip(np.digitize(z, zedges) - 1, 0, n_bins - 1)

    # Initialize result DataFrame
    result = sf_df.copy()

    result["scaling"] = scaling[xidx, yidx, zidx]
    result["scaling_lower"] = scaling_lower[xidx, yidx, zidx]
    result["scaling_upper"] = scaling_upper[xidx, yidx, zidx]

    return result


def correct_sf(interval, correction_lookup):
    """
    Correct the structure function using the correction lookup table.
    Also smooth the scaling function prior to applying the correction.
    This is done to avoid discontinuities in the correction factor.

    Parameters:
    - sf: DataFrame containing the structure function data
    - correction_lookup: DataFrame containing the correction factors

    Returns:
    - Corrected structure function
    """

    if interval["gap_status"] != "corrected":
        raise ValueError(
            "This function is only for correcting structure functions with gap_status 'corrected'."
        )

    sf_df = pd.DataFrame(
        {
            "lag": interval["lag"],
            "sf": interval["sf"],
            "gp": interval["gp"],
        }
    )

    # Merge the SF with the correction lookup table
    sf_df_scaled = compute_scaling(
        sf_df,
        correction_lookup,
    )

    scaling_smooth = smooth_scaling(sf_df_scaled.lag, sf_df_scaled.scaling)
    scaling_smooth_lower = smooth_scaling(sf_df_scaled.lag, sf_df_scaled.scaling_lower)
    scaling_smooth_upper = smooth_scaling(sf_df_scaled.lag, sf_df_scaled.scaling_upper)

    # Apply the correction factor to the SF
    corrected_sf = sf_df_scaled.sf * scaling_smooth
    corrected_sf_lower = sf_df_scaled.sf * scaling_smooth_lower
    corrected_sf_upper = sf_df_scaled.sf * scaling_smooth_upper

    return corrected_sf.values, corrected_sf_lower.values, corrected_sf_upper.values


# Load data
v1_raw = pd.read_pickle("../data/interim/voyager/voyager1_lism.pkl")
v2_raw = pd.read_pickle("../data/interim/voyager/voyager2_lism.pkl")
print("Data loaded successfully.")

v1_ls = v1_raw[["BR", "BT", "BN"]].resample("288s").mean()
v1_ls_lint = v1_ls.interpolate(method="linear")

nlags = 100
lags_v1_ls = np.logspace(0, np.log10(0.25 * len(v1_ls)), nlags)
# Convert array to integers
lags_v1_ls = [int(x) for x in lags_v1_ls]
# Drop duplicate lags
lags_v1_ls = np.unique([int(x) for x in lags_v1_ls])

# Compute SF for full 11 years, using fast SF
sf_v1_ls = sf_funcs.compute_sf_fast(
    v1_ls,
    lags_v1_ls,
    [2],
    return_missing=False,
)
sf_v1_ls_lint = sf_funcs.compute_sf_fast(
    v1_ls_lint,
    lags_v1_ls,
    [2],
    return_missing=False,
)
print("Structure function for Voyager 1 large scale data computed successfully.")

v1_ss = v1_ls.loc["2020-01-01":"2020-07-01"]
v1_ss_lint = v1_ss.interpolate(method="linear")

lags_v1_ss = np.logspace(0, np.log10(0.25 * len(v1_ss)), nlags)
# Convert array to integers
lags_v1_ss = [int(x) for x in lags_v1_ss]
# Drop duplicate lags
lags_v1_ss = np.unique([int(x) for x in lags_v1_ss])

# Compute SF for one 180 day interval, using fast SF
sf_v1_ss, sf_v1_ss_missing = sf_funcs.compute_sf_fast(
    v1_ss,
    lags_v1_ss,
    [2],
    return_missing=True,
)
sf_v1_ss_lint = sf_funcs.compute_sf_fast(
    v1_ss_lint,
    lags_v1_ss,
    [2],
)
print(
    "Structure function for Voyager 1 small scale data (180 days) computed successfully."
)

n_bins = 25
with open(f"../results/full/correction_lookup_3d_{n_bins}_bins_lint.pkl", "rb") as f:
    correction_lookup = pickle.load(f)


sf_v1_ss_df = pd.DataFrame(
    {
        "lag": lags_v1_ss,
        "sf": sf_v1_ss,
        "gp": sf_v1_ss_missing,
    }
)

# Correct the small-scale SF
sf_v1_ss_corr, sf_v1_ss_corr_lower, sf_v1_ss_corr_upper = correct_sf(
    sf_v1_ss_df,
    correction_lookup,
)


# Overlay
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(lags_v1_ls, sf_v1_ls, label="Naive (Full)", color="red", lw=5, alpha=0.3)
ax.loglog(lags_v1_ss, sf_v1_ss, label="Naive (180 days)", color="red", lw=1)
ax.loglog(
    lags_v1_ls, sf_v1_ls_lint, label="LINT (Full)", color="purple", lw=5, alpha=0.3
)
ax.loglog(lags_v1_ss, sf_v1_ss_lint, label="LINT (180 days)", color="purple", lw=1)
ax.loglog(lags_v1_ss, sf_v1_ss_corr, label="Corrected (180 days)", color="black", lw=1)
ax.set_title("V1 VLISM Structure Functions")
ax.set_xlabel("Lag (s)")
ax.set_ylabel("Structure Function")
ax.legend()
plt.show()

# -> At large scales, different methods look the same (LINT vs naive)
# -> At small scales, different methods are different, but different interval lengths are the same

# Now do standardised versions
