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
run_mode = params.run_mode
n_bins_list = params.n_bins_list

# Set matplotlib font size
plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


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
                # Make some interim plots for inspection (full publication versions are done
                # in script 4c locally)
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
                    f"Distribution of missing proportion and lag ({gap_handling})",
                    y=1.1,
                )
                ax.set_facecolor("black")
                ax.set_xscale("log")
                plt.savefig(
                    f"results/{run_mode}/plots/train_heatmap_{n_bins}bins_2d_{gap_handling}.pdf",
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
                    f"results/{run_mode}/plots/train_heatmap_{n_bins}bins_2d_counts.pdf",
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

                # Define the number of columns (you can adjust this as desired)
                n_cols = 5  # Number of columns per row
                n_rows = (
                    n_bins + n_cols - 1
                ) // n_cols  # Calculate number of rows needed

                # MISSING VS LAG, BY POWER BIN
                fig, ax = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(n_cols * 1.5, n_rows * 1.7),
                    sharex=True,
                    sharey=True,
                )
                plt.subplots_adjust(wspace=0.18, hspace=0.5)
                plt.grid(False)
                plt.suptitle(
                    r"3D error heatmap: trend with increasing $\mathbf{power}$",
                    y=0.98,
                )

                # Flatten the axis array to simplify indexing
                ax = ax.flatten()
                for i in range(n_bins):
                    ax[i].grid(False)
                    c = ax[i].pcolormesh(
                        xedges,
                        yedges,
                        pe_mean[:, :, i],
                        cmap="bwr",
                    )
                    c.set_clim(-100, 100)
                    ax[i].set_title(
                        f"({np.round(zedges[i], 2)},{np.round(zedges[i+1], 2)})",
                    )
                    ax[i].set_facecolor("black")
                    ax[i].semilogx()

                fig.text(
                    0.5,
                    0.03,
                    "Lag ($\\tau$)",
                    ha="center",
                    va="center",
                )  # Shared x-axis label
                fig.text(
                    0.05,
                    0.5,
                    "\% missing",
                    ha="center",
                    va="center",
                    rotation="vertical",
                )  # Shared y-axis label

                # Hide any extra subplots if n_bins is not a multiple of n_cols
                for j in range(n_bins, len(ax)):
                    fig.delaxes(ax[j])

                # Add a color bar on the right-hand side of the figure, stretching down the entire height
                cbar_ax = fig.add_axes(
                    [0.92, 0.105, 0.02, 0.78]
                )  # [left, bottom, width, height] to cover full height
                cb = plt.colorbar(
                    c, cax=cbar_ax
                )  # Attach the color bar to the last heatmap
                cb.set_label("MPE")  # Optional: Label the color bar

                plt.savefig(
                    f"results/{run_mode}/plots/train_heatmap_{n_bins}bins_3d_{gap_handling}_power.pdf",
                    bbox_inches="tight",
                )
                plt.close()

                # POWER VS % MISSING, BY LAG BIN
                fig, ax = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(n_cols * 1.5, n_rows * 1.7),
                    sharex=True,
                    sharey=True,
                )
                plt.subplots_adjust(wspace=0.18, hspace=0.5)
                plt.grid(False)
                plt.suptitle(
                    r"3D error heatmap: trend with increasing $\mathbf{lag}$",
                    y=0.98,  # Was 1.02 for 2 rows
                )

                # Flatten the axis array to simplify indexing
                ax = ax.flatten()

                # Format lag bin edges to integers
                formatted_xedges = [f"{x:.0f}".rstrip("0").rstrip(".") for x in xedges]
                for i in range(n_bins):
                    ax[i].grid(False)
                    c = ax[i].pcolormesh(
                        yedges,
                        zedges,
                        pe_mean[i, :, :],
                        cmap="bwr",
                    )
                    c.set_clim(-100, 100)
                    ax[i].set_title(
                        f"({formatted_xedges[i]},{formatted_xedges[i+1]})",
                    )
                    ax[i].set_facecolor("black")
                    ax[i].semilogy()
                fig.text(
                    0.5, 0.03, "\% missing", ha="center", va="center"
                )  # Shared x-axis label, was 0.00 y-val for 2 rows
                fig.text(
                    0.05,
                    0.5,
                    "Power",
                    ha="center",
                    va="center",
                    rotation="vertical",
                )  # Shared y-axis label

                # Hide any extra subplots if n_bins is not a multiple of n_cols
                for j in range(n_bins, len(ax)):
                    fig.delaxes(ax[j])

                # Add a color bar on the right-hand side of the figure, stretching down the entire height
                cbar_ax = fig.add_axes(
                    [0.92, 0.105, 0.02, 0.78]
                )  # [left, bottom, width, height] to cover full height
                cb = plt.colorbar(
                    c, cax=cbar_ax
                )  # Attach the color bar to the last heatmap
                cb.set_label("MPE")  # Optional: Label the color bar

                plt.savefig(
                    f"results/{run_mode}/plots/train_heatmap_{n_bins}bins_3d_{gap_handling}_lag.pdf",
                    bbox_inches="tight",
                )
                plt.close()

                # POWER VS LAG, BIN % MISSING BIN
                fig, ax = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(n_cols * 1.5, n_rows * 1.7),
                    sharex=True,
                    sharey=True,
                )
                plt.subplots_adjust(wspace=0.18, hspace=0.5)
                plt.grid(False)
                plt.suptitle(
                    r"3D error heatmap: trend with increasing \% $\mathbf{missing}$",
                    y=0.98,
                )
                # Flatten the axis array to simplify indexing
                ax = ax.flatten()
                for i in range(n_bins):
                    ax[i].grid(False)
                    c = ax[i].pcolormesh(
                        xedges,
                        zedges,
                        pe_mean[:, i, :],
                        cmap="bwr",
                    )
                    c.set_clim(-100, 100)
                    ax[i].set_title(
                        f"({np.round(yedges[i], 2)},{np.round(yedges[i+1], 2)})",
                    )
                    ax[i].set_facecolor("black")
                    ax[i].semilogx()
                    ax[i].semilogy()
                fig.text(
                    0.5, 0.03, "Lag ($\\tau$)", ha="center", va="center"
                )  # Shared x-axis label
                fig.text(
                    0.05,
                    0.5,
                    "Power",
                    ha="center",
                    va="center",
                    rotation="vertical",
                )  # Shared y-axis label

                # Hide any extra subplots if n_bins is not a multiple of n_cols
                for j in range(n_bins, len(ax)):
                    fig.delaxes(ax[j])

                # Add a color bar on the right-hand side of the figure, stretching down the entire height
                cbar_ax = fig.add_axes(
                    [0.92, 0.105, 0.02, 0.78]
                )  # [left, bottom, width, height] to cover full height
                cb = plt.colorbar(
                    c, cax=cbar_ax
                )  # Attach the color bar to the last heatmap
                cb.set_label("MPE")  # Optional: Label the color bar

                plt.savefig(
                    f"results/{run_mode}/plots/train_heatmap_{n_bins}bins_3d_{gap_handling}_missing.pdf",
                    bbox_inches="tight",
                )
                plt.close()

            if dim == 3 and gap_handling == "naive":
                pass
            else:
                # Export the lookup tables as a pickle file
                # WE NEED TO DO THIS FOR NAIVE AS WELL, FOR PLOTS
                output_file_path = f"results/{run_mode}/correction_lookup_{dim}d_{n_bins}_bins_{gap_handling}.pkl"
                with open(
                    output_file_path,
                    "wb",
                ) as f:
                    pickle.dump(correction_lookup, f)
                print(f"Saved complete correction lookup table {output_file_path}")
