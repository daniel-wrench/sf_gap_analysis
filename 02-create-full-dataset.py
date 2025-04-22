import glob

import pandas as pd
import seaborn as sns

# Read in the scalar stats from all files
csv_file_list = sorted(glob.glob("data/processed/psp/*_scalar_stats.csv"))
print(f"Reading in {len(csv_file_list)} files")

df = pd.concat(
    [pd.read_csv(csv, index_col=False) for csv in csv_file_list], ignore_index=True
)
print(f"Successfully read in and concatenated {len(csv_file_list)} files")
duration = pd.to_datetime(df["end_time"].values[-1]) - pd.to_datetime(
    df["start_time"].values[0]
)
print(
    f"Combined dataframe is {duration} long, from {df.start_time[0]} to {df.end_time[0]}"
)

# Limit to original intervals
df = df[df.gap_status == "original"]

# Compute derived scalars (see reynolds script: process_data.py)

df["dboB0_mean"] = df["db_mean"] / df["B0_mean"]
# df["Re_lt"] = df["tce"] / df["ttc"]

# pickle.dump(scalar_and_vector_stats)
# pd.to_csv("scalar_stats.csv")

# PLOT RESULTS (likely want this in a separate script, but check works on subset here first)

# df = pd.read_csv("scalar_stats.csv", parse_dates=["timestamp"])
# df = df.set_index("timestamp")

# Remove metadata columns for statistical analysis
df_study = df.iloc[:, 9:]
df_study.describe()
df_study.corr()

sns.pairplot(df_study.iloc[:500, :4], diag_kind="kde", plot_kws={"alpha": 0.2})

# plt.savefig("pairplot.png")
