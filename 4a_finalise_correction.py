##########################
# SERIAL JOB TO COMBINE ALL RESULTS

import glob
import pickle

import numpy as np
from matplotlib import pyplot as plt

import src.params as params

spacecraft = "psp"
gap_handling = "lint"
missing_measure = "missing_percent"
data_path_prefix = params.data_path_prefix
output_path = params.output_path
n_bins_list = params.n_bins_list

for gap_handling in ["lint", "naive"]:
    for dim in [2, 3]:
        for n_bins in n_bins_list:
            max_lag = params.int_length * params.max_lag_prop
            xedges = (
                np.logspace(0, np.log10(max_lag), n_bins + 1) - 0.01
            )  # so that first lag bin starts just before 1
            xedges[-1] = max_lag + 1
            yedges = np.linspace(0, 100, n_bins + 1)  # Missing prop
            zedges = np.logspace(-2, 1, n_bins + 1)  # ranges from 0.01 to 10

            # Read in all pe arrays from data/processed and combine them in a list
            pe_list = []
            for file in glob.glob(
                f"{data_path_prefix}data/processed/psp/train/errors/*pe_{dim}d_{n_bins}_bins_{gap_handling}.pkl"
            ):  # LIMIT HERE!!
                with open(file, "rb") as f:
                    pe_list.append(pickle.load(f))
                    print(f"Loaded {file}")
            print(f"Loaded {len(pe_list)} files")
            if dim == 2:
                pe_mean = np.full((n_bins, n_bins), fill_value=np.nan)
                pe_min = np.full((n_bins, n_bins), fill_value=np.nan)
                pe_max = np.full((n_bins, n_bins), fill_value=np.nan)
                pe_std = np.full((n_bins, n_bins), fill_value=np.nan)
                n = np.full((n_bins, n_bins), fill_value=np.nan)

                for i in range(n_bins):
                    for j in range(n_bins):
                        all_pe = [
                            array[i, j]
                            for array in pe_list
                            if not np.all(np.isnan(array[i, j]))
                        ]
                        if len(all_pe) != 0:
                            all_pe = np.concatenate(all_pe)
                            pe_mean[i, j] = np.nanmean(all_pe)
                            pe_std[i, j] = np.nanstd(all_pe)
                            pe_min[i, j] = np.nanmin(all_pe)
                            pe_max[i, j] = np.nanmax(all_pe)
                            n[i, j] = len(all_pe)

                scaling = 1 / (1 + pe_mean / 100)
                scaling_lower = 1 / (1 + (pe_mean + 2 * pe_std) / 100)
                scaling_upper = 1 / (1 + (pe_mean - 2 * pe_std) / 100)

                scaling[np.isnan(scaling)] = 1
                scaling_lower[np.isnan(scaling_lower)] = 1
                scaling_upper[np.isnan(scaling_upper)] = 1

                # Export these arrays in an efficient manner
                correction_lookup = {
                    "xedges": xedges,
                    "yedges": yedges,
                    "scaling": scaling,
                    "scaling_lower": scaling_lower,
                    "scaling_upper": scaling_upper,
                    "pe_mean": pe_mean,
                    "pe_std": pe_std,
                    "pe_min": pe_min,
                    "pe_max": pe_max,
                    "n": n,
                }

                fig, ax = plt.subplots(figsize=(7, 5))
                plt.grid(False)
                plt.pcolormesh(
                    xedges,
                    yedges,
                    pe_mean.T,
                    cmap="bwr",
                )
                plt.grid(False)
                plt.colorbar(label="MPE")
                plt.clim(-100, 100)
                plt.xlabel("Lag ($\\tau$)")
                plt.ylabel("Missing percentage")
                plt.title(
                    f"Distribution of missing proportion and lag ({gap_handling.upper()})",
                    y=1.1,
                )
                ax.set_facecolor("black")
                ax.set_xscale("log")
                plt.savefig(
                    f"plots/results/{output_path}/train_heatmap_{n_bins}bins_2d.pdf",
                    bbox_inches="tight",
                )
                plt.close()

                fig, ax = plt.subplots(figsize=(7, 5))
                plt.grid(False)
                plt.pcolormesh(
                    xedges,
                    yedges,
                    n.T,
                    cmap="Greens",
                )
                # Remove gridlines
                plt.grid(False)
                plt.colorbar(label="Count of lag pairs")
                plt.xlabel("Lag ($\\tau$)")
                plt.ylabel("Missing percentage")
                plt.title("Distribution of missing proportion and lag", y=1.1)
                ax.set_facecolor("black")
                ax.set_xscale("log")
                plt.savefig(
                    f"plots/results/{output_path}/train_heatmap_{n_bins}bins_2d_counts.pdf",
                    bbox_inches="tight",
                )
                plt.close()

            elif (dim == 3) and (gap_handling == "lint"):  # has zedges too
                pe_mean = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)
                pe_min = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)
                pe_max = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)
                pe_std = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)
                n = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)

                for i in range(n_bins):
                    for j in range(n_bins):
                        for k in range(n_bins):
                            all_pe = [
                                array[i, j, k]
                                for array in pe_list
                                if not np.all(np.isnan(array[i, j, k]))
                            ]
                            if len(all_pe) != 0:
                                all_pe = np.concatenate(all_pe)
                                pe_mean[i, j, k] = np.nanmean(all_pe)
                                pe_std[i, j, k] = np.nanstd(all_pe)
                                pe_min[i, j, k] = np.nanmin(all_pe)
                                pe_max[i, j, k] = np.nanmax(all_pe)
                                n[i, j, k] = len(all_pe)

                scaling = 1 / (1 + pe_mean / 100)
                scaling_lower = 1 / (1 + (pe_mean + 1 * pe_std) / 100)
                scaling_upper = 1 / (1 + (pe_mean - 1 * pe_std) / 100)

                scaling[np.isnan(scaling)] = 1
                scaling_lower[np.isnan(scaling_lower)] = 1
                scaling_upper[np.isnan(scaling_upper)] = 1

                # Export these arrays in an efficient manner
                correction_lookup = {
                    "xedges": xedges,
                    "yedges": yedges,
                    "zedges": zedges,
                    "scaling": scaling,
                    "scaling_lower": scaling_lower,
                    "scaling_upper": scaling_upper,
                    "pe_mean": pe_mean,
                    "pe_std": pe_std,
                    "pe_min": pe_min,
                    "pe_max": pe_max,
                    "n": n,
                }

                fig, ax = plt.subplots(
                    1, n_bins, figsize=(n_bins * 3, 3.5), tight_layout=True
                )
                # Remove spacing between subplots
                plt.subplots_adjust(wspace=0.2)
                plt.grid(False)
                for i in range(n_bins):
                    ax[i].grid(False)
                    c = ax[i].pcolormesh(
                        xedges,
                        yedges,
                        pe_mean[:, :, i],
                        cmap="bwr",
                    )
                    # plt.colorbar(label="MPE")
                    c.set_clim(-100, 100)
                    plt.xlabel("Lag ($\\tau$)")
                    plt.ylabel("Missing proportion")
                    plt.title("Distribution of missing proportion and lag")
                    ax[i].set_facecolor("black")
                    ax[i].semilogx()
                    ax[i].set_title(
                        f"Power bin {i+1}/{n_bins}".format(np.round(zedges[i], 2))
                    )
                    ax[i].set_xlabel("Lag ($\\tau$)")
                    # Remove y-axis labels for all but the first plot
                    if i > 0:
                        ax[i].set_yticklabels([])
                        ax[i].set_ylabel("")

                plt.savefig(
                    f"plots/results/{output_path}/train_heatmap_{n_bins}bins_3d_{gap_handling.upper()}_power.pdf",
                    bbox_inches="tight",
                )
                plt.close()

                fig, ax = plt.subplots(
                    1, n_bins, figsize=(n_bins * 3, 3.5), tight_layout=True
                )
                # Remove spacing between subplots
                plt.grid(False)
                plt.subplots_adjust(wspace=0.2)
                for i in range(n_bins):
                    ax[i].grid(False)
                    c = ax[i].pcolormesh(
                        yedges,
                        zedges,
                        pe_mean[i, :, :],
                        cmap="bwr",
                    )
                    # plt.colorbar(label="MPE")
                    c.set_clim(-100, 100)
                    ax[i].set_xlabel("Missing prop")
                    ax[i].set_ylabel("Power")
                    plt.title("Distribution of missing proportion and lag")
                    ax[i].set_facecolor("black")
                    ax[i].semilogy()
                    ax[i].set_title(
                        f"Lag bin {i+1}/{n_bins}".format(np.round(zedges[i], 2))
                    )
                    ax[i].set_xlabel("Missing prop")
                    # Remove y-axis labels for all but the first plot
                    if i > 0:
                        ax[i].set_yticklabels([])
                        ax[i].set_ylabel("")

                plt.savefig(
                    f"plots/results/{output_path}/train_heatmap_{n_bins}bins_3d_{gap_handling.upper()}_lag.pdf",
                    bbox_inches="tight",
                )
                plt.close()

                fig, ax = plt.subplots(
                    1, n_bins, figsize=(n_bins * 3, 3.5), tight_layout=True
                )
                # Remove spacing between subplots
                plt.grid(False)
                plt.subplots_adjust(wspace=0.2)
                for i in range(n_bins):
                    ax[i].grid(False)
                    c = ax[i].pcolormesh(
                        xedges,
                        zedges,
                        pe_mean[:, i, :],
                        cmap="bwr",
                    )
                    # plt.colorbar(label="MPE")
                    c.set_clim(-100, 100)
                    plt.title("Distribution of missing proportion and lag")
                    ax[i].set_facecolor("black")
                    ax[i].semilogx()
                    ax[i].semilogy()
                    ax[i].set_title(
                        f"Missing prop bin {i+1}/{n_bins}".format(
                            np.round(zedges[i], 2)
                        )
                    )
                    ax[i].set_xlabel("Lag ($\\tau$)")
                    ax[i].set_ylabel("Power")
                    # Remove y-axis labels for all but the first plot
                    if i > 0:
                        ax[i].set_yticklabels([])
                        ax[i].set_ylabel("")

                plt.savefig(
                    f"plots/results/{output_path}/train_heatmap_{n_bins}bins_3d_{gap_handling.upper()}_missing.pdf",
                    bbox_inches="tight",
                )
                plt.close()

            if dim == 3 and gap_handling == "naive":
                pass
            else:
                # Export the lookup tables as a pickle file
                # WE NEED TO DO THIS FOR NAIVE AS WELL, FOR PLOTS
                output_file_path = f"data/corrections/{output_path}/correction_lookup_{dim}d_{n_bins}_bins_{gap_handling}.pkl"
                with open(
                    output_file_path,
                    "wb",
                ) as f:
                    pickle.dump(correction_lookup, f)
                print(f"Saved complete correction lookup table {output_file_path}")
