import os
import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/")

# Generate fake data for Velocity vs. Time
def generate_velocity_data(num_points=250):
    time = np.linspace(0, 100, num_points)
    raw_velocity = np.sin(time) + np.random.normal(0, 0.5, num_points)
    sifted_velocity = np.sin(time)
    df = pd.DataFrame({'Time': time, 'Raw Velocity': raw_velocity, 'Sifted Velocity': sifted_velocity})
    return df

# Generate fake data for Arrival Times
def generate_arrival_times_data(num_points=250):
    event_id = np.arange(1, num_points + 1)
    true_arrival_times = np.sort(np.random.uniform(0, 1000, num_points))
    predicted_arrival_times = true_arrival_times + np.random.normal(0, 20, num_points)
    df = pd.DataFrame({
        'Event ID': event_id,
        'True Arrival Time': true_arrival_times,
        'Predicted Arrival Time': predicted_arrival_times
    })
    return df

# Create Velocity vs. Time plot
def create_velocity_vs_time_plot():
    df = generate_velocity_data()
    fig = px.line(
        df,
        x='Time',
        y=['Raw Velocity', 'Sifted Velocity'],
        labels={'value': 'Velocity', 'variable': 'Data Type'},
        title='Velocity vs. Time for Raw and Sifted Data'
    )
    fig.update_layout(title={'x': 0.5}, legend_title_text='Data Type')
    fig.for_each_trace(
        lambda t: t.update(line=dict(dash='solid' if t.name == 'Sifted Velocity' else 'dot'))
    )
    return fig

# Create Arrival Times plot
def create_arrival_times_plot():
    df = generate_arrival_times_data()
    fig = px.scatter(
        df,
        x='True Arrival Time',
        y='Predicted Arrival Time',
        labels={
            'True Arrival Time': 'True Arrival Time',
            'Predicted Arrival Time': 'Predicted Arrival Time'
        },
        title='True Arrival Times vs. Predicted Arrival Times',
        opacity=0.7
    )
    fig.update_layout(title={'x': 0.5}, legend_title_text='Legend')
    # Add ideal prediction line y=x
    fig.add_shape(
        type='line',
        x0=df['True Arrival Time'].min(),
        y0=df['True Arrival Time'].min(),
        x1=df['True Arrival Time'].max(),
        y1=df['True Arrival Time'].max(),
        line=dict(color='Red', dash='dash'),
        name='Ideal Prediction'
    )
    # Bring the ideal line to back
    fig['data'] = fig['data'][1:] + fig['data'][:1]
    return fig

# Layout with improved design
layout = dbc.Container([
    # Header Section
    dbc.Row([
        dbc.Col([
            html.H1("S.I.F.T.E.R.", className="display-3"),
            html.H4("Seismic Investigation and Frequency Tracking for Extraterrestrial Research",
                    className="text-muted"),
            html.Hr(className="my-2"),
        ], width=12, className='text-center')
    ], className="my-4"),

    # Introduction Section
    dbc.Row([
        dbc.Col([
            html.P(
                """
                S.I.F.T.E.R. is a comprehensive project designed to analyze seismic data from planetary missions, 
                particularly the Apollo and Mars InSight Lander. The primary goal is to detect seismic events 
                (e.g., quakes) and distinguish them from noise using machine learning models and advanced seismic 
                processing techniques. This project integrates data from lunar, Martian, and Earth-based seismic 
                sources, optimizing the system for deployment on non-terrestrial seismometers. The system is 
                initially developed in Python for rapid prototyping and machine learning, then finalized in C++ 
                for deployment in low-power environments on planetary missions. A terrestrial front-end is included 
                for real-time analysis and visualization, leveraging tools such as Streamlit, Plotly, or Dash.
                """,
                className="lead"
            )
        ], width=10)
    ], className="mb-5 justify-content-center"),

    # Visualization Section
    dbc.Row([
        dbc.Col([
            html.Label("Select a Visualization to View", className="h5"),
            dcc.Dropdown(
                id='graph-dropdown',
                options=[
                    {'label': 'Velocity vs. Time', 'value': 'velocity_time'},
                    {'label': 'True vs. Predicted Arrival Times', 'value': 'arrival_times'}
                ],
                value='velocity_time',
                clearable=False
            ),
        ], width=6, className='mx-auto')
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='graph'),
            html.P(id='dynamic-text', className="mt-3")
        ], width=10, className='mx-auto')
    ], className="mb-5"),

    # About Section
    dbc.Row([
        dbc.Col([
            html.H2("About Our Model", className="my-4"),
            html.P(
                "Our model leverages state-of-the-art machine learning algorithms to analyze seismic data with unprecedented accuracy and efficiency."
            )
        ], width=10)
    ], className="justify-content-center")
], fluid=True)

# Callback to update graph and text
@callback(
    [Output('graph', 'figure'),
     Output('dynamic-text', 'children')],
    [Input('graph-dropdown', 'value')]
)
def update_graph(value):
    if value == 'velocity_time':
        plotly_fig = create_velocity_vs_time_plot()
        text = "This visualization represents velocity over time for raw and sifted seismic data."
    elif value == 'arrival_times':
        plotly_fig = create_arrival_times_plot()
        text = "This visualization compares true arrival times with predicted arrival times."
    else:
        plotly_fig = {}
        text = "No visualization selected."
    return plotly_fig, text
