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
from scipy.interpolate import interp1d

# Fit a power law to the corrected SF
from scipy.optimize import curve_fit

import src.params as params
import src.sf_funcs as sfs
import src.utils as utils

# Set sans-serif font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# Read in cleaned Voyager 1 data
df = pd.read_pickle("data/interim/voyager/voyager2_lism.pkl")
print("Loaded dataset")

interval_days = 170


tau_min = 5
tau_max = 20
cadence = 48

df_res = df.resample(str(cadence) + "s").mean()

# Calculate the total timespan of the data
start_date = df.index.min()
end_date = df.index.max()

# Create interval boundaries
interval_start = start_date
interval_boundaries = []

while interval_start <= end_date:
    interval_end = interval_start + pd.Timedelta(days=interval_days)
    interval_boundaries.append((interval_start, interval_end))
    interval_start = interval_end


# Initialize the results DataFrame
results_df = pd.DataFrame()

# Process each interval
for start, end in interval_boundaries:
    # Get data in this interval (inclusive of start, exclusive of end)
    mask = (df_res.index >= start) & (df_res.index < end)
    int_norm = utils.normalize(df_res[mask])
    bad_input = int_norm[["BR", "BT", "BN"]]

    # Skip empty intervals
    if bad_input.empty:
        continue

    # Compute mean for this interval
    interval_mean = bad_input.mean()

    # Compute SF for this interval

    bad_output = sfs.compute_sf(bad_input, np.arange(1, 100), [2], False, False)

    # Get ACF from SF
    # var_signal = np.sum(np.var(input, axis=0))
    var_signal = 3
    # will always be this variance as we are using the standardised 3D SF
    acf_from_sf = 1 - (bad_output.sf_2 / (2 * var_signal))
    try:
        ttu, taylor_scale_u_std = utils.compute_taylor_chuychai(
            bad_output["lag"].values,
            acf_from_sf.values,
            tau_min=tau_min,
            tau_max=tau_max,
        )
    except:
        print(f"Error in compute_taylor_chuychai for interval {start} to {end}")
        ttu = np.nan
        continue

    # Add the start timestamp and bad_output to the results dictionary
    new_row = pd.DataFrame(
        {
            "start": start,
            "end": end,
            "lags_s": [bad_output.lag * cadence],
            "sf_2": [bad_output.sf_2],
            "acf_from_sf": [acf_from_sf],
            "ttu_s": ttu * cadence,
        },
        index=[0],
    )

    # Append the new row to the DataFrame

    results_df = pd.concat([results_df, new_row], ignore_index=True)


# Plot all the sf_2
fig, ax = plt.subplots(figsize=(8, 5))
# Plot the SF for each interval
for i, row in results_df.iterrows():
    # Plot the SF
    ax.plot(row["lags_s"], row["acf_from_sf"])
    if i == 0:
        ax.scatter(
            row["lags_s"][tau_min:tau_max],
            row["acf_from_sf"][tau_min:tau_max],
            color="black",
            marker="x",
            s=3,
            zorder=10,
            label=r"$\lambda_T$ max lag range",
        )
plt.legend()
# Set the y-axis to be between 0 and 1
# ax.set_ylim(0.9999, 1.0001)
# # Set the x-axis to be between 0 and 100
ax.set_xlim(0, tau_max * cadence * 2)

print(f"Parameters: {tau_min=}, {tau_max=}, {cadence=}")

results_df["ttu_hours"] = results_df["ttu_s"] / 3600
results_df[["ttu_hours", "ttu_s"]].round(2)
results_df["ttu_hours"].describe()

# Export the results as a CSV file
results_df.to_csv("results/full/voyager2_lism_ttu_hr.csv", index=False)

results_df.head()
