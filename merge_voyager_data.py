import glob

import pandas as pd
from sunpy.timeseries import TimeSeries

import src.params as params

# Define spacecraft and their start years
spacecraft_config = {"voyager1": 2009, "voyager2": 2015}

cadence = "48s"

for spacecraft, start_year in spacecraft_config.items():
    print(f"\n=== Processing {spacecraft.upper()} ===")

    # Load and concatenate all CDF files for the spacecraft
    file_pattern = f"data/raw/voyager/{spacecraft}_{cadence}_mag-vim_*.cdf"
    print(f"Looking for files matching: {file_pattern}")

    files = glob.glob(file_pattern)
    if not files:
        print(f"Warning: No files found for {spacecraft}")
        continue

    print(f"Found {len(files)} files for {spacecraft}")

    try:
        data = TimeSeries(files, concatenate=True)
        df_raw = data.to_dataframe()[str(start_year) :]
    except Exception as e:
        print(f"Error loading TimeSeries data for {spacecraft}: {e}")
        continue

    # Get magnetic field components + orbital radius
    vars = params.mag_vars_dict["voyager"].copy()
    vars.append("Radius")
    df_raw = df_raw.loc[:, vars]

    df = df_raw.resample(cadence).mean()

    print(f"\n{spacecraft.upper()} Data Info:")
    print(df.info())
    print(f"\n{spacecraft.upper()} Data Head:")
    print(df.head())

    # Save to pickle file
    output_file = f"data/interim/voyager/{spacecraft}_hs_lism.pkl"
    df.to_pickle(output_file)
    print(f"Exported merged {spacecraft} data to {output_file}")

print("\n=== Processing Complete ===")
