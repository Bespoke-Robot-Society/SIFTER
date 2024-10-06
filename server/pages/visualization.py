import os
import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input
from matplotlib import pyplot as plt
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from plotly.tools import mpl_to_plotly
dash.register_page(__name__, path="/")
styles = {"padding": "20px"}

# Create a simple Matplotlib figure
def create_matplotlib_figure1():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [10, 20, 25, 30], label='Example')
    ax.set_title('Matplotlib to Plotly')
    ax.legend()
    return fig

def create_matplotlib_figure2():
    fig, ax = plt.subplots()
    ax.plot([2, 3, 4, 5], [20, 30, 35, 40], label='Example')
    ax.set_title('Matplotlib to Plotly')
    ax.legend()
    return fig

# Convert Matplotlib figure to Plotly
def convert_to_plotly(fig):
    plotly_fig = mpl_to_plotly(fig)
    return plotly_fig

simple_plotly_fig1 = convert_to_plotly(create_matplotlib_figure1())
simple_plotly_fig2 = convert_to_plotly(create_matplotlib_figure2())
layout = html.Div(
    [
        html.H3("S.I.F.T.E.R."),
        html.H5(
            "Seismic Investigation and Frequency Tracking for Extraterrestrial Research"
        ),
        html.Br(),
        html.P(
            """S.I.F.T.E.R. is a comprehensive project designed to analyze seismic data from planetary missions, particularly the Apollo and Mars InSight Lander. The primary goal is to detect seismic events (e.g., quakes) and distinguish them from noise using machine learning models and advanced seismic processing techniques. This project integrates data from lunar, Martian, and Earth-based seismic sources, optimizing the system for deployment on non-terrestrial seismometers. The system is initially developed in Python for rapid prototyping and machine learning, then finalized in C++ for deployment in low-power environments on planetary missions. A terrestrial front-end is included for real-time analysis and visualization, leveraging tools such as Streamlit, Plotly, or Dash."""
        ),
        dbc.Container(
            dbc.Container(
                [
                html.Label("Select a visual to see"),
                dcc.Dropdown(
                id='graph-dropdown',
                options=[
                    {'label': 'option1', 'value': 'option1'},
                    {'label': 'option2', 'value': 'option2'}
                ],
                value='option1',  # Default value
                clearable=False,
                style={'width': '50%'}
                )
                ]
            ),
            style={
            'display': 'flex',        # Align items in a row
            'justify-content': 'space-around',  # Distribute items evenly
            'align-items': 'center'  # Vertically align items
            }
        ),
    html.Br(),
    dbc.Container(
        [
        dcc.Graph(id='matplotlib-graph'),
        html.P("Testing to see how this looks on the page!")
        ],
        style={
        'display': 'flex',        # Align items in a row
        'justify-content': 'space-around',  # Distribute items evenly
        'align-items': 'center'  # Vertically align items
        }
    ),
    html.H2("About our model"),
    html.P("Our model is the greatest model that ever did model")
    ],
    style=styles,
)
# Callback to update graph1
@callback(
    Output('matplotlib-graph', 'figure'),
    [Input('graph-dropdown', 'value')]
)
def update_graph(value):
    if value == 'option1':
        plotly_fig = simple_plotly_fig1
    else:
        plotly_fig = simple_plotly_fig2
    return plotly_fig
