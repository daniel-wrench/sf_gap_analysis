import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read in the scalar stats from all files
spacecraft = "wind"

csv_file_list = sorted(glob.glob(f"data/processed/{spacecraft}/*_scalar_stats.csv"))
df = pd.concat(
    [pd.read_csv(csv, index_col=False) for csv in csv_file_list[:3]], ignore_index=True
)
print(f"Successfully read in and concatenated {len(csv_file_list)} files")
duration = pd.to_datetime(df["end_time"].values[-1]) - pd.to_datetime(
    df["start_time"].values[0]
)
print(
    f"Combined dataframe consists of {len(df)} rows across {duration}, \nfrom {df.start_time.values[0]} to {df.end_time.values[-1]}"
)

# Compute derived scalars (see reynolds script: process_data.py)

df["dboB0_mean"] = df["db_mean"] / df["B0_mean"]
# df["tce_s"] = df["tce"] * df["cadence"].str.rstrip("s").astype(float)
# df["ttu_s"] = df["ttu"] * df["cadence"].str.rstrip("s").astype(float)
# df["tce_km"]
# df["Re_lt"] = df["tce"] / df["ttc"]

# pickle.dump(scalar_and_vector_stats)
# pd.to_csv("scalar_stats.csv")

# PLOT RESULTS (likely want this in a separate script, but check works on subset here first)

# df = pd.read_csv("scalar_stats.csv", parse_dates=["timestamp"])
# df = df.set_index("timestamp")

# Limit to true intervals
df = df[df.gap_status == "true"]

# Remove metadata columns for statistical analysis
df_study = df.iloc[:, 9:]
df_study["gap_status"] = df["gap_status"]
print("\nSummary stats:\n")
print(df_study.describe())

# print("\nCorrelation matrix:\n")
# print(df_study.corr())

sns.pairplot(
    df_study.iloc[:500, :],
    vars=["tce", "ttu", "qi_sf"],
    hue="gap_status",
    corner=True,
    diag_kind="kde",
    plot_kws={"alpha": 0.2},
)
plt.show()
# See further customisations here:
# https://seaborn.pydata.org/generated/seaborn.pairplot.html

# plt.savefig("pairplot.png")
