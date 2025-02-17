# THINGS TO FIX
# Option to have shared or not y-axis
# (A bit problematic. Could somewhat solve using zoom_state to retain zoom when clicked)
# Better labelling of derived stats
# True time series behind LINT
# More metadata: % removed next to time series, error values next to SFs
# USE TO GET BETTER CASE STUDIES FOR PAPER

import glob
import pickle

import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html

import src.params as params

# CASE STUDY PLOTS
# Pre-correction case studies


plt.rc("text", usetex=True)
plt.rc("font", family="serif", serif="Computer Modern", size=10)
plt.rcParams.update(
    {
        "font.size": 10,  # Set font size to match LaTeX (e.g., 10pt)
        "axes.labelsize": 10,  # Label size
        "xtick.labelsize": 10,  # X-axis tick size
        "ytick.labelsize": 10,  # Y-axis tick size
        "legend.fontsize": 10,  # Legend font size
        "figure.titlesize": 10,  # Figure title size
        "figure.dpi": 300,  # Higher resolution figure output
    }
)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

np.random.seed(123)  # For reproducibility

times_to_gap = params.times_to_gap

# Import all corrected (test) files
spacecraft = "wind"
n_bins = 25
# times_to_gap = params.times_to_gap # removing as will only be using this file locally

data_path_prefix = params.data_path_prefix
run_mode = params.run_mode
pwrl_range = params.pwrl_range

index = 0
# 2  # For now, just getting first corrected file
# NOTE: THIS IS NOT THE SAME AS FILE INDEX!
# due to train-test split, file indexes could be anything

# Importing one corrected file

print(f"Calculating stats for {spacecraft} data with {n_bins} bins")
input_file_list = sorted(
    glob.glob(
        f"results/{run_mode}/test_sfs_corrected_subset/wi_*_corrected_{n_bins}_bins_with_sfs.pkl"
    )
)
print(f"About to read {len(input_file_list)} files")

all_files_metadata = []
all_ints_metadata = []
all_ints = []
all_ints_gapped_metadata = []
all_ints_gapped = []
all_sfs_gapped_corrected = []

for file in input_file_list[:5]:
    try:
        with open(file, "rb") as f:
            data = pickle.load(f)
    except pickle.UnpicklingError:
        print(f"UnpicklingError encountered in file: {file}.")
        continue
    except EOFError:
        print(f"EOFError encountered in file: {file}.")
        continue
    except Exception as e:
        print(f"An unexpected error {e} occurred with file: {file}.")
        continue
    print(f"Loaded {file}")

    # Unpack the dictionary and append to lists
    all_files_metadata.append(data["files_metadata"])
    all_ints_metadata.append(data["ints_metadata"])
    all_ints.append(data["ints"])
    all_ints_gapped_metadata.append(data["ints_gapped_metadata"])
    all_ints_gapped.append(data["ints_gapped"])
    all_sfs_gapped_corrected.append(data["sfs_gapped_corrected"])

# Concatenate all dataframes
files_metadata = pd.concat(all_files_metadata, ignore_index=True)
ints_metadata = pd.concat(all_ints_metadata, ignore_index=True)
ints_gapped_metadata = pd.concat(all_ints_gapped_metadata, ignore_index=True)
ints_gapped = pd.concat(all_ints_gapped, ignore_index=True)
sfs_gapped_corrected = pd.concat(all_sfs_gapped_corrected, ignore_index=True)

# Flatten the list of lists for ints
ints = [item for sublist in all_ints for item in sublist]

print(
    f"Successfully read in {input_file_list[index]}. This contains {len(ints_metadata)}x{times_to_gap} intervals"
)

# Why does file index 4, int_index 1, v15 not have a LINT version?
# Same with 3/0/8
# Is it removed by the following filtering

# Check sample size counts


# Filtering out bad original tces, setting gap-affected ones to max when cannot be found
ints_gapped_metadata = ints_gapped_metadata[ints_gapped_metadata.tce_orig >= 0]
ints_gapped_metadata.loc[ints_gapped_metadata.tce == -1, "tce"] = (
    params.max_lag_prop * params.int_length
)
# sfs_gapped_corrected = sfs_gapped_corrected[
#     sfs_gapped_corrected.gap_handling != "corrected_2d"
# ]

# Only getting relevant columns from particularly large dfs
ints_gapped = ints_gapped[
    ["time", "file_index", "int_index", "version", "gap_handling", "Bx"]
]
sfs_gapped_corrected = sfs_gapped_corrected[
    ["file_index", "int_index", "version", "lag", "sf_2", "gap_handling"]
]

ints_gapped_metadata = ints_gapped_metadata[
    [
        "file_index",
        "int_index",
        "version",
        "gap_handling",
        "missing_percent_overall",
        "missing_percent_chunks",
        "mape",
        "slope_ape",
        "tce_ape",
        "ttu_ape",
    ]
]

# Convert ins_gapped_metadata to long form for the scatter plot
ints_gapped_metadata_long = ints_gapped_metadata.melt(
    id_vars=[
        "file_index",
        "int_index",
        "version",
        "gap_handling",
        "missing_percent_overall",
        "missing_percent_chunks",
    ],
    value_vars=["mape", "slope_ape", "tce_ape", "ttu_ape"],
    var_name="derived_stat",
    value_name="ape",
)


def create_faceted_scatter(selected_criteria=None, independent_yaxes=False):
    """
    Create a faceted scatter plot.

    If selected_criteria is provided (a tuple: (file_index, int_index, version)),
    update the marker color and size for points matching that criteria.
    """
    # Create a copy so we do not modify the original df
    df_local = ints_gapped_metadata_long.copy()

    # Set default marker color and size
    df_local["marker_size"] = 3
    df_local["marker_symbol"] = "circle"
    df_local["marker_color"] = df_local["gap_handling"]

    if selected_criteria is not None:
        file_index, int_index, version = selected_criteria
        # Create a mask that matches the selected criteria
        mask = (
            (df_local["file_index"] == file_index)
            & (df_local["int_index"] == int_index)
            & (df_local["version"] == version)
        )
        # Change the marker properties for the matching rows
        df_local.loc[mask, "marker_color"] = "purple"
        df_local.loc[mask, "marker_size"] = 15
        df_local.loc[mask, "marker_symbol"] = "x"

    # Build the scatterplot with facets
    fig = px.scatter(
        df_local,
        x="missing_percent_overall",
        y="ape",
        color="marker_color",
        # Specify colours
        color_discrete_map={
            "lint": "black",
            "naive": "indianred",
            "corrected_3d": "#1b9e77",
            "corrected_3d_smoothed": "purple",
            "corrected_2d": "#d95f02",
            "purple": "purple",
        },
        # Map for symbols
        symbol_map={"x": "x", "circle": "circle"},
        opacity=0.2,
        size="marker_size",
        size_max=15,
        symbol="marker_symbol",
        facet_col="gap_handling",
        facet_row="derived_stat",
        hover_data=[
            "missing_percent_overall",
            # "ape",
            "file_index",
            "int_index",
            "version",
        ],
        # title="Metadata Scatter Plot",
    )

    # Remove the legend for color since we're using it just for highlighting
    fig.update_yaxes(matches=None if independent_yaxes else "y")  # Toggle shared y-axis
    fig.update_layout(showlegend=False)
    fig.update_traces(marker=dict(line=dict(width=0)))  # Remove border around points

    return fig


# Add a global variable for shared y-axis toggle
independent_yaxes = False

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Interactively Exploring SF Errors From Gapped Time Series"),
        # Need to pivot to long form for the dropdown
        # dcc.Dropdown(
        #     id="error-metric",
        #     options=[
        #         {"label": col, "value": col}
        #         for col in ["file_index", "int_index", "version"]
        #     ],
        #     value="file_index",  # Default selection
        #     clearable=False,
        # ),
        dcc.Checklist(
            id="toggle-ind-yaxes",
            options=[{"label": "Independent Y-Axes", "value": "independent"}],
            value=[],  # Default to non-independent (shared)
            inline=True,
        ),
        dcc.Graph(
            id="scatter-plot",
            figure=create_faceted_scatter(),
            style={"height": "100vh"},
        ),
        html.Div(
            id="selected-info",
            style={
                "marginTop": "5px",
                "fontWeight": "bold",
                "fontSize": "20px",
            },
        ),
        html.Hr(style={"margin": "0px"}),
        html.Div(
            [
                dcc.Graph(
                    id="ts-plot", figure={}, style={"height": "60vh", "width": "100%"}
                ),
                dcc.Graph(
                    id="sf-plot", figure={}, style={"height": "60vh", "width": "100%"}
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "justifyContent": "center",
            },
        ),
    ],
    style={"padding": "0px"},
)


@app.callback(
    [Output("ts-plot", "figure"), Output("sf-plot", "figure")],
    Input("scatter-plot", "clickData"),
)
def update_line_plots(clickData):
    if clickData is None:
        default_ts_fig = px.line(
            title="Click a point on the scatter plot to see its time series"
        )
        default_sf_fig = px.line(
            title="Click a point on the scatter plot to see its structure function"
        )
        return default_ts_fig, default_sf_fig

    # Extract the clicked point's data
    # Note: The structure of clickData depends on how your figure is set up.
    # Here we assume the customdata/hover_data includes the three fields.
    point = clickData["points"][0]
    # Depending on how the data was attached, these keys might be in point['customdata'] or point directly.
    # For this example, let's assume they are in customdata in the same order as hover_data.
    file_index = point["customdata"][0]
    int_index = point["customdata"][1]
    version = point["customdata"][2]

    # Filter the time series dataframe
    mask_ts = (
        (ints_gapped["file_index"] == file_index)
        & (ints_gapped["int_index"] == int_index)
        & (ints_gapped["version"] == version)
        & (ints_gapped["gap_handling"] == "lint")
    )

    df_ts = ints_gapped[mask_ts]

    ts_fig = px.line(
        df_ts,
        x="time",
        y="Bx",
        color_discrete_sequence=["grey"],
        # title=f"TS for file_index: {file_index}, int_index: {int_index}, version: {version}",
    )

    # Filter the time series dataframe
    mask_sf = (
        (sfs_gapped_corrected["file_index"] == file_index)
        & (sfs_gapped_corrected["int_index"] == int_index)
        & (sfs_gapped_corrected["version"] == version)
    )

    df_sf = sfs_gapped_corrected[mask_sf]

    sf_fig = px.line(
        df_sf,
        x="lag",
        y="sf_2",
        color="gap_handling",
        color_discrete_map={
            "naive": "indianred",
            "lint": "black",
            "corrected_3d": "#1b9e77",
            "corrected_3d_smoothed": "purple",
            "corrected_2d": "#d95f02",
            "true": "grey",
        },
        log_x=True,
        log_y=True,
        # title=f"SF for file_index: {file_index}, int_index: {int_index}, version: {version}",
    )
    # Add horizontal line at 3.8
    sf_fig.add_hline(
        y=3.8,
        line_dash="dot",
        line_color="black",
        label=dict(
            text="1/e equiv.",
            textposition="end",
            font=dict(size=12, color="black"),
            yanchor="top",
        ),
    )

    sf_fig.add_vrect(
        x0=params.tau_min,
        x1=params.tau_max,
        fillcolor="green",
        opacity=0.1,
        label=dict(
            text="TS fit range",
            textposition="top center",
        ),
        line_width=0,
    )

    sf_fig.add_vrect(
        x0=params.pwrl_range[0],
        x1=params.pwrl_range[1],
        fillcolor="salmon",
        opacity=0.1,
        label=dict(
            text="Slope fit range",
            textposition="top center",
        ),
        line_width=0,
    )

    return ts_fig, sf_fig


@app.callback(
    [Output("scatter-plot", "figure"), Output("selected-info", "children")],
    [Input("scatter-plot", "clickData"), Input("toggle-ind-yaxes", "value")],
)
def update_highlight(clickData, independent_yaxes_value):

    independent_yaxes = "independent" in independent_yaxes_value

    if clickData is None:
        return create_faceted_scatter(), "No point selected yet."

    # Extract the clicked point's data.
    # Note: Depending on your plot, the structure of clickData may vary.
    # Here, we assume that hover_data values are available in customdata.
    point = clickData["points"][0]
    # The order of customdata corresponds to the order of hover_data we provided.
    file_index = point["customdata"][0]
    int_index = point["customdata"][1]
    version = point["customdata"][2]

    selected_info = f"Selected point -> file_index: {file_index}, int_index: {int_index}, version: {version}"

    # Create a new figure with highlighted points.
    fig = create_faceted_scatter(
        (file_index, int_index, version), independent_yaxes=independent_yaxes
    )

    return fig, selected_info


if __name__ == "__main__":
    app.run_server(debug=True)
