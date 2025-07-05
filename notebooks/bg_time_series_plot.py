import matplotlib.pyplot as plt
import pandas as pd

# Set font size and inward ticks
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in"})
plt.rcParams.update({"font.size": 14})

v1_hp_date = pd.to_datetime("2012-08-25")
v2_hp_date = pd.to_datetime("2018-11-05")

v1_raw = pd.read_pickle("../data/interim/voyager/voyager1_hs_lism.pkl")
v1_res = v1_raw.resample("1d").mean()[v1_hp_date:"2024-01-01"]
v1_res.info()

v2_raw = pd.read_pickle("../data/interim/voyager/voyager2_hs_lism.pkl")
v2_res = v2_raw.resample("1d").mean()[v2_hp_date:]
v2_res.info()

# Change index to Radius variable
v1_res.index = v1_res["Radius"]
v2_res.index = v2_res["Radius"]

# Plot 2-week moving averages of mag for each spacecraft
for spacecraft in ["v1", "v2"]:
    data = eval(f"{spacecraft}_res")

    if spacecraft == "v1":
        width = 8
        xlim = 121.8, 163
    elif spacecraft == "v2":
        width = 3
        xlim = 119.5, 133

    fig, axes = plt.subplots(figsize=(width, 1.6))
    data["F1"].rolling(window=14, center=True).mean().plot(
        color=colors[spacecraft], linewidth=2, ax=axes
    )
    # Remove spines
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["bottom"].set_visible(False)

    axes.set_ylabel("$|\mathbf{B}|$ (nT)", fontsize=15)
    axes.set_xlabel("Distance (au)", fontsize=15)

    # Increase font size for y-axis tick labels and set to white
    axes.tick_params(axis="y", labelsize=14)

    axes.set_xlim(*xlim)

    # plt.show()
    plt.savefig(
        f"../bg_figs/bg_{spacecraft}_vlism.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )

v2_raw_hr = v2_raw.resample("1h").mean()
v2_raw_hr.set_index("Radius", inplace=True)
v2_raw_sub = v2_raw_hr.loc[127:130]
# Standardize the data
# v2_raw_sub_hr = (v2_raw_sub_hr - v2_raw_sub_hr.mean()) / v2_raw_sub_hr.std()


v2_raw_sub[["BR", "BT", "BN"]].plot(figsize=(8, 3))
# Remove legend
plt.legend().set_visible(False)

# v2_raw_sub = v2_raw.loc["2021-08-06":"2014-08-09"]
v2_raw_sub["BT"].plot(figsize=(12, 3), color="tab:orange")
plt.show()
