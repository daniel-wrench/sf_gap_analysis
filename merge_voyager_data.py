import glob

import pandas as pd
from sunpy.timeseries import TimeSeries

import src.params as params

spacecraft = "voyager2"

if spacecraft == "voyager1":
    start_year = 2011
elif spacecraft == "voyager2":
    start_year = 2017

cadence = "48s"

data = TimeSeries(
    # f"../data/raw/voyager/{spacecraft}_{cadence}_mag-vim_{year}0101_v01.cdf",
    glob.glob(f"data/raw/voyager/{spacecraft}_{cadence}_mag-vim_*.cdf"),
    concatenate=True,
)

df_raw = data.to_dataframe()[str(start_year) :]

# Get magnetic field components + orbital radius
vars = params.mag_vars_dict["voyager"].copy()
vars.append("Radius")
df_raw = df_raw.loc[:, vars]

df = df_raw.resample(cadence).mean()

print(df.info())

print(df.head())

df.to_pickle(f"data/interim/voyager/{spacecraft}_hs_lism.pkl")
print(
    f"Exported merged {spacecraft} data to data/interim/voyager/{spacecraft}_hs_lism.pkl"
)
