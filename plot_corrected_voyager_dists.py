import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set matplotlib styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in"})


def load_and_process_voyager_data():
    """Load and process Voyager 1 and 2 data"""
    # Voyager 1 data
    df_v1 = pd.read_csv("results/full/voyager1_corrected_metadata_NEW_RANGE.csv")
    df_v1["tce_s"] = df_v1["tce"] * df_v1["cadence"]
    df_v1["ttu_s"] = df_v1["ttu"] * df_v1["cadence"]
    df_v1["tce_days"] = df_v1["tce_s"] / (24 * 3600)
    df_v1["ttu_hours"] = df_v1["ttu_s"] / 3600
    df_v1["source"] = "Voyager 1"

    # Voyager 2 data
    df_v2 = pd.read_csv("results/full/voyager2_corrected_metadata_NEW_RANGE.csv")
    df_v2["tce_s"] = df_v2["tce"] * df_v2["cadence"]
    df_v2["ttu_s"] = df_v2["ttu"] * df_v2["cadence"]
    df_v2["tce_days"] = df_v2["tce_s"] / (24 * 3600)
    df_v2["ttu_hours"] = df_v2["ttu_s"] / 3600
    df_v2["source"] = "Voyager 2"

    # High resolution Voyager 1 data
    high_res_v1 = pd.read_csv("results/full/voyager1_lism_ttu_hr.csv")
    high_res_v1["source"] = "Voyager 1"

    return pd.concat([df_v1, df_v2], ignore_index=True), high_res_v1


def create_fraternale_data():
    """Create Fraternale 2021 reference data"""
    return pd.DataFrame(
        {
            "tce_days": [63.5, 19, 76, 19, 60, np.nan, np.nan, np.nan],
            "ttu_hours": [0.048, 0.06, 0.07, 0.054, 0.063, np.nan, np.nan, np.nan],
            "slope": [0.36, 0.28, 0.27, 0.29, 0.67, 0.48, 0.37, 0.72],
            "source": ["Fraternale et al."] * 8,
        }
    )


def plot_histograms():
    """Create the main histogram plots"""
    # Load data
    df_combined, high_res_v1 = load_and_process_voyager_data()
    fraternale_data = create_fraternale_data()

    # Plot configuration
    vars_to_plot = ["slope", "tce_days", "ttu_hours"]
    labels = [r"$\beta$", r"$\lambda_C$ (days)", r"$\lambda_T$ (hours)"]
    bin_edges = {
        "slope": np.linspace(0.1, 0.9, 12),
        "tce_days": np.linspace(0.3, 70, 10),
        "ttu_hours": np.linspace(0, 0.08, 12),
    }

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(5, 2), sharey=True)

    # Color palette
    colors = {"Voyager 1": "#2ca02c", "Voyager 2": "#1f77b4"}

    for i, var in enumerate(vars_to_plot):
        ax = axes[i]

        # Select appropriate data source
        if var == "ttu_hours":
            plot_data = high_res_v1
        else:
            plot_data = df_combined[df_combined["gap_handling"] == "corrected_3d"]

        # Create histogram
        sns.histplot(
            plot_data,
            x=var,
            hue="source",
            hue_order=["Voyager 1", "Voyager 2"],
            palette=colors,
            ax=ax,
            bins=bin_edges[var],
            element="step",
            legend=False,  # We'll handle legend separately
        )

        # Add Fraternale reference points as stars
        if var in fraternale_data.columns:
            frat_values = fraternale_data[var].dropna()
            if len(frat_values) > 0:
                ax.scatter(
                    frat_values,
                    np.full_like(frat_values, 0.8),
                    marker="*",
                    s=100,
                    color="#7c7c7c28",
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.4,
                    zorder=5,
                )

        # Customize axes
        ax.set_xlim(bin_edges[var][0], bin_edges[var][-1])
        ax.set_xlabel(labels[i], fontsize=10)

        # Add K41 reference line for slope
        if var == "slope":
            ax.axvline(2 / 3, color="black", linestyle="dotted", alpha=0.6)
            ax.text(2 / 3 + 0.01, 6, "K41", fontsize=10, alpha=0.6)

        # Print statistics
        print(
            f"95% CI for {var}: "
            f"{plot_data[var].quantile(0.025):.2f}, "
            f"{plot_data[var].quantile(0.975):.2f}, "
            f"median: {plot_data[var].median():.2f}"
        )

    # Set consistent y-axis limits
    axes[0].set_ylabel("Count", fontsize=10)
    max_y = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, max_y)

    # Special x-axis adjustment for middle plot
    axes[1].set_xlim(-5, 80)

    # Create unified legend
    create_legend(axes[1])

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.8, wspace=0.1)

    # Save figure
    plt.savefig(
        "results/full/plots/voyager/voyager1_2_corrected_stats_NEW_RANGE.png",
        dpi=300,
        bbox_inches="tight",
    )

    return fig


def create_legend(ax):
    """Create a unified legend with all plot elements"""
    # Create legend handles
    legend_elements = [
        mlines.Line2D([0], [0], color="#2ca02c8b", lw=4, label="Voyager 1"),
        mlines.Line2D([0], [0], color="#1f76b4a7", lw=4, label="Voyager 2"),
        mlines.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#7c7c7c28",
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=10,
            label="Previous V1 estimates",
            linestyle="None",
        ),
    ]

    # Add legend
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.4),
        ncol=3,
        frameon=False,
        fontsize=8,
    )


if __name__ == "__main__":
    fig = plot_histograms()
    print("Plot completed successfully!")
