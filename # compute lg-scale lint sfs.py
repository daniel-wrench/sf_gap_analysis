# compute lg-scale lint sfs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.sf_funcs as sf_funcs
import src.utils as utils

v1_raw = pd.read_pickle("data/interim/voyager/voyager1_lism.pkl")
v1_raw.info()

v2_raw = pd.read_pickle("data/interim/voyager/voyager2_lism.pkl")
v2_raw.info()

v1_raw_lint = v1_raw.interpolate(method="linear")
v2_raw_lint = v2_raw.interpolate(method="linear")

lags_v1 = np.logspace(3.3, np.log10(0.25 * v1_raw_lint.shape[0]), 100)
lags_v2 = np.logspace(3.3, np.log10(0.25 * v2_raw_lint.shape[0]), 100)

sf_v1 = sf_funcs.compute_sf_fast(v1_raw_lint[["BR", "BT", "BN"]], lags_v1, [2])
sf_v2 = sf_funcs.compute_sf_fast(v2_raw_lint[["BR", "BT", "BN"]], lags_v2, [2])

plt.loglog(lags_v1 * 48, sf_v1, label="Voyager 1")
plt.loglog(lags_v2 * 48, sf_v2, label="Voyager 2")
plt.xlabel("Lag (s)")
plt.ylabel("Structure Function")
plt.legend()
plt.show()

voyager_sf_data = {
    "V1": {"lags": lags_v1, "sf": sf_v1},
    "V2": {"lags": lags_v2, "sf": sf_v2},
}
import pickle

with open("data/interim/voyager/vlism_sfs_short.pkl", "wb") as f:
    pickle.dump(voyager_sf_data, f)
