# CASE STUDY PLOTS
# Pre-correction case studies

import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
import src.params as params
import src.data_import_funcs as dif


np.random.seed(123)  # For reproducibility

times_to_gap = params.times_to_gap

plt.rc("font", family="serif", serif=["Computer Modern Roman"], size=10)
plt.rc("text", usetex=True)

# Import all corrected (test) files
spacecraft = "wind"
n_bins = int(sys.argv[1])
# times_to_gap = params.times_to_gap # removing as will only be using this file locally

data_path_prefix = params.data_path_prefix
output_path = params.output_path
pwrl_range = params.pwrl_range

index = 1  # For now, just getting first corrected file
# NOTE: THIS IS NOT THE SAME AS FILE INDEX!
# due to train-test split, file indexes could be anything

# Importing one corrected file

print(f"Calculating stats for {spacecraft} data with {n_bins} bins")
if spacecraft == "psp":
    input_file_list = sorted(
        glob.glob(
            data_path_prefix
            + f"data/corrections/{output_path}/psp_*_corrected_{n_bins}_bins_FULL.pkl"
        )
    )
elif spacecraft == "wind":
    input_file_list = sorted(
        glob.glob(
            data_path_prefix
            + f"data/corrections/{output_path}/wi_*_corrected_{n_bins}_bins_FULL.pkl"
        )
    )
else:
    raise ValueError("Spacecraft must be 'psp' or 'wind'")

file = input_file_list[index]
try:
    with open(file, "rb") as f:
        data = pickle.load(f)
except pickle.UnpicklingError:
    print(f"UnpicklingError encountered in file: {file}.")
except EOFError:
    print(f"EOFError encountered in file: {file}.")
except Exception as e:
    print(f"An unexpected error {e} occurred with file: {file}.")
print(f"Loaded {file}")

# Unpack the dictionary
files_metadata = data["files_metadata"]
ints_metadata = data["ints_metadata"]
ints = data["ints"]
ints_gapped_metadata = data["ints_gapped_metadata"]
ints_gapped = data["ints_gapped"]
sfs = data["sfs"]
sfs_gapped_corrected = data["sfs_gapped_corrected"]

print(
    f"Successfully read in {input_file_list[index]}. This contains {len(ints_metadata)}x{times_to_gap} intervals"
)

# Parameters for the case study plots

file_index = files_metadata["file_index"].values[0]
# THERE IS NOW JUST ONE FILE INDEX TO WORRY ABOUT
# (so could really remove this indexing from code below)

int_index = 0  # Selecting first (possibly only) interval from this file

print(
    "Currenting making plots for file index",
    file_index,
    "and interval",
    int_index,
)

fig, ax = plt.subplots(2, 3, figsize=(15, 2 * 3), sharey="col")
# will use consistent interval index, but choose random versions of it to plot
versions_to_plot = [
    16,
    21,
    8,
    # 23,  # 13 = 9% missing, 23 = 50% missing
    # 8,
]
for ax_index, version in enumerate(versions_to_plot):
    if len(ints) == 1:
        ax[ax_index, 0].plot(
            ints[0]["Bx"].values, c="grey"
        )  # Just plotting one component for simplicity
    else:
        ax[ax_index, 0].plot(
            ints[0][int_index][0]["Bx"].values, c="grey"
        )  # Just plotting one component for simplicity
    # Not currently plotting due to indexing issue: need to be able to index
    # on both file_index and int_index
    ax[ax_index, 0].plot(
        ints_gapped.loc[
            (ints_gapped["file_index"] == file_index)
            & (ints_gapped["int_index"] == int_index)
            & (ints_gapped["version"] == version)
            & (ints_gapped["gap_handling"] == "lint"),
            "Bx",  # Just plotting one component for simplicity
        ].values,
        c="black",
    )

    # Put missing_percent_overall in the title
    ax[ax_index, 0].set_title(
        f"{ints_gapped_metadata.loc[(ints_gapped_metadata['file_index']==file_index) & (ints_gapped_metadata['int_index']==int_index) & (ints_gapped_metadata['version']==version) & (ints_gapped_metadata['gap_handling']=='lint'), 'missing_percent_overall'].values[0]:.1f}\% missing"
    )

    # Plot the SF
    ax[ax_index, 1].plot(
        sfs.loc[
            (sfs["file_index"] == file_index) & (sfs["int_index"] == int_index),
            "lag",
        ],
        sfs.loc[
            (sfs["file_index"] == file_index) & (sfs["int_index"] == int_index),
            "sf_2",
        ],
        c="grey",
        label="True",
        lw=4,
    )

    ax[ax_index, 1].plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "sf_2",
        ],
        c="indianred",
        label="Naive ({:.1f})".format(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
                & (ints_gapped_metadata["gap_handling"] == "naive"),
                "mape",
            ].values[0]
        ),
    )

    ax[ax_index, 1].plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "lint"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "lint"),
            "sf_2",
        ],
        c="black",
        label="LINT ({:.1f})".format(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
                & (ints_gapped_metadata["gap_handling"] == "lint"),
                "mape",
            ].values[0]
        ),
    )

    # Plot the sf_2_pe
    ax[ax_index, 2].plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "lint"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "lint"),
            "sf_2_pe",
        ],
        c="black",
    )
    ax[ax_index, 2].plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "sf_2_pe",
        ],
        c="indianred",
    )

    # plot sample size n on right axis
    ax2 = ax[ax_index, 2].twinx()
    ax2.plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "missing_percent",
        ],
        c="grey",
        # linestyle="--",
        lw=1,
    )

    # Label the axes
    ax[1, 0].set_xlabel("Time")
    ax[ax_index, 0].set_ylabel("$B_R$ (normalised)")
    ax[1, 1].set_xlabel("Lag ($\\tau$)")
    ax[ax_index, 1].set_ylabel("SF")
    ax[1, 2].set_xlabel("Lag ($\\tau$)")
    ax[ax_index, 2].set_ylabel("\% error")
    ax2.set_ylabel("\% pairs missing", color="grey")
    ax2.tick_params(axis="y", colors="grey")
    ax2.set_ylim(0, 100)

    ax[ax_index, 2].axhline(0, c="black", linestyle="--")
    ax[ax_index, 2].set_ylim(-100, 100)

    ax[ax_index, 1].set_xscale("log")
    ax[ax_index, 1].set_yscale("log")
    ax[ax_index, 2].set_xscale("log")
    ax[ax_index, 1].legend()
    [ax[0, i].set_xticklabels([]) for i in range(3)]

# Add titles
ax[0, 1].set_title("Structure function estimates")
ax[0, 2].set_title("SF \% error and \% pairs missing")
plt.subplots_adjust(wspace=0.4)

plt.savefig(
    f"plots/results/{output_path}/test_{spacecraft}_case_study_gapping_{file_index}_{int_index}.pdf",
    bbox_inches="tight",
)

# 5e. Corrected case studies

fig, axs = plt.subplots(figsize=(12, 4), ncols=3, sharey=True, sharex=True)
plt.subplots_adjust(wspace=0.1)


for ax_index, version in enumerate(versions_to_plot):
    ax = axs[ax_index]

    ax.plot(
        sfs[(sfs["file_index"] == file_index) & (sfs["int_index"] == int_index)]["lag"],
        sfs[(sfs["file_index"] == file_index) & (sfs["int_index"] == int_index)][
            "sf_2"
        ],
        color="black",
        label="True",
        lw=4,
        alpha=0.5,
    )
    ax.plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "naive"),
            "sf_2",
        ],
        color="red",
        lw=1,
        label="Naive ({:.1f})".format(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
                & (ints_gapped_metadata["gap_handling"] == "naive"),
                "mape",
            ].values[0]
        ),
    )
    ax.plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "lint"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "lint"),
            "sf_2",
        ],
        color="black",
        lw=1,
        label="LINT ({:.1f})".format(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
                & (ints_gapped_metadata["gap_handling"] == "lint"),
                "mape",
            ].values[0]
        ),
    )
    # ax.plot(
    #     sfs_gapped_corrected.loc[
    #         (sfs_gapped_corrected["file_index"] == file_index)
    #         & (sfs_gapped_corrected["int_index"] == int_index)
    #         & (sfs_gapped_corrected["version"] == version)
    #         & (sfs_gapped_corrected["gap_handling"] == "corrected_2d"),
    #         "lag",
    #     ],
    #     sfs_gapped_corrected.loc[
    #         (sfs_gapped_corrected["file_index"] == file_index)
    #         & (sfs_gapped_corrected["int_index"] == int_index)
    #         & (sfs_gapped_corrected["version"] == version)
    #         & (sfs_gapped_corrected["gap_handling"] == "corrected_2d"),
    #         "sf_2",
    #     ],
    #     color="blue",
    #     lw=1,
    #     label="Corrected (2D) ({:.1f})".format(
    #         ints_gapped_metadata.loc[
    #             (ints_gapped_metadata["file_index"] == file_index)
    #             & (ints_gapped_metadata["int_index"] == int_index)
    #             & (ints_gapped_metadata["version"] == version)
    #             & (ints_gapped_metadata["gap_handling"] == "corrected_2d"),
    #             "mape",
    #         ].values[0]
    #     ),
    # )
    ax.plot(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
            "sf_2",
        ],
        color="#1b9e77",
        lw=1,
        label="Corrected ({:.1f})".format(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
                & (ints_gapped_metadata["gap_handling"] == "corrected_3d"),
                "mape",
            ].values[0]
        ),
    )
    ax.fill_between(
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
            "lag",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
            "sf_2_lower",
        ],
        sfs_gapped_corrected.loc[
            (sfs_gapped_corrected["file_index"] == file_index)
            & (sfs_gapped_corrected["int_index"] == int_index)
            & (sfs_gapped_corrected["version"] == version)
            & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
            "sf_2_upper",
        ],
        color="#1b9e77",
        alpha=0.2,
    )

    missing = ints_gapped_metadata.loc[
        (ints_gapped_metadata["file_index"] == file_index)
        & (ints_gapped_metadata["int_index"] == int_index)
        & (ints_gapped_metadata["version"] == version),
        "missing_percent_overall",
    ].values

    ax.legend(loc="lower right")
    ax.semilogx()
    ax.semilogy()

    # PLOTTING HEATMAP IN FIRST PANEL

    # Label test intervals with letters
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    ax.annotate(
        f"{alphabet[ax_index]}: {float(missing[0]):.1f}\% missing",
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0.1, 0.9),
        textcoords="axes fraction",
        transform=ax.transAxes,
        c="black",
        bbox=dict(
            facecolor="lightgrey", edgecolor="white", boxstyle="round", alpha=0.7
        ),
    )

    ax.set_xlabel("Lag ($\\tau$)")

    print(f"\nStats for interval version {alphabet[ax_index]} as plotted:\n")
    print(
        ints_gapped_metadata.loc[
            (ints_gapped_metadata["file_index"] == file_index)
            & (ints_gapped_metadata["int_index"] == int_index)
            & (ints_gapped_metadata["version"] == version)
        ][
            [
                "file_index",
                "int_index",
                "version",
                "missing_percent_overall",
                "gap_handling",
                "mape",
                "slope_pe",
            ]
        ]
    )

    # Plot the slope of the SF
    slope_corrected = ints_gapped_metadata.loc[
        (ints_gapped_metadata["file_index"] == file_index)
        & (ints_gapped_metadata["int_index"] == int_index)
        & (ints_gapped_metadata["version"] == version)
        & (ints_gapped_metadata["gap_handling"] == "corrected_3d"),
        "slope",
    ].values[0]

    dif.pltpwrl(
        pwrl_range[0],
        0.1,
        pwrl_range[0],
        pwrl_range[1],
        slope_corrected,
        lw=1,
        ls="--",
        color="#1b9e77",
        label=f"Log-log slope: {slope_corrected:.3f}",
        ax=ax,
    )
    # Plot the slope of the SF
    slope_naive = ints_gapped_metadata.loc[
        (ints_gapped_metadata["file_index"] == file_index)
        & (ints_gapped_metadata["int_index"] == int_index)
        & (ints_gapped_metadata["version"] == version)
        & (ints_gapped_metadata["gap_handling"] == "naive"),
        "slope",
    ].values[0]

    dif.pltpwrl(
        pwrl_range[0],
        0.1,
        pwrl_range[0],
        pwrl_range[1],
        slope_naive,
        lw=1,
        ls="--",
        color="red",
        label=f"Log-log slope: {slope_corrected:.3f}",
        ax=ax,
    )
    # Plot the slope of the SF
    slope = ints_metadata.loc[
        (ints_metadata["file_index"] == file_index)
        & (ints_metadata["int_index"] == int_index),
        "slope",
    ].values[0]

    dif.pltpwrl(
        pwrl_range[0],
        0.1,
        pwrl_range[0],
        pwrl_range[1],
        slope,
        lw=2,
        ls="--",
        color="grey",
        label=f"Log-log slope: {slope_corrected:.3f}",
        ax=ax,
    )

plt.savefig(
    f"plots/results/{output_path}/test_{spacecraft}_case_study_correcting_{file_index}_{int_index}_{n_bins}_bins.pdf",
    bbox_inches="tight",
)
