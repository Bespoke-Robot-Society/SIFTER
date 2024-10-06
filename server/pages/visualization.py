import os
import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
dash.register_page(__name__, path="/")
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')
styles = {"padding": "20px"}
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

image_folder = 'assets/'
model_image_folder = 'assets/images/model_output/'
# List of available images (assuming these are already saved)
lunar_image_files = {
    '1': 'lunar_preprocessed_images/xa.s12.00.mhz.1970-01-19HR00_evid00002.mseed_combined.png',
    '2': 'lunar_preprocessed_images/xa.s12.00.mhz.1970-03-25HR00_evid00003.mseed_combined.png',
    '3': 'lunar_preprocessed_images/xa.s12.00.mhz.1970-03-26HR00_evid00004.mseed_combined.png'
}
mars_image_files = {
    '1': 'martian_preprocessed_images/XB.ELYSE.02.BHV.2022-01-02HR04_evid0006.mseed_combined.png',
    '2': 'martian_preprocessed_images/XB.ELYSE.02.BHV.2022-02-03HR08_evid0005.mseed_combined.png'
}
test_image_files = {
    '1': 'test_images/S12_GradeB/xa_combined.png'
}
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
        dcc.Dropdown(
        id='graph-dropdown',
        options=[
            {'label': 'lunar_image1', 'value': 'lunar_image1'},
            {'label': 'lunar_image2', 'value': 'lunar_image2'},
            {'label': 'lunar_image3', 'value': 'lunar_image3'},
            {'label': 'mars_image1', 'value': 'mars_image1'},
            {'label': 'mars_image2', 'value': 'mars_image2'},
            {'label': 'mars_image3', 'value': 'mars_image3'},
            {'label': 'test_image1', 'value': 'test_image1'}
        ],
        value='line',  # Default value
        clearable=False,
        style={'width': '50%'}
    ),
    html.Br(),
    # Div to display the selected graph image
    html.Img(id='image-display', style={'width': '70%', 'height': 'auto'})
    ],
    style=styles,
)

# Define the callback to update the image based on button click
@callback(
    Output('image-display', 'src'),
    [Input('graph-dropdown', 'value')]
)
def display_image(value):
    ctx = dash.callback_context

    # Default image is image1
    img_file = lunar_image_files['1']
    
    if ctx.triggered:
        
        if value == 'lunar_image1':
            img_file = lunar_image_files['1']
        elif value == 'lunar_image2':
            img_file = lunar_image_files['2']
        elif value == 'lunar_image3':
            img_file = lunar_image_files['3']
        elif value == 'mars_image1':
            img_file = mars_image_files['1']
        elif value == 'mars_image2':
            img_file = mars_image_files['2']
        elif value == 'test_image1':
            img_file = test_image_files['1']
    # Construct the path to the image file
    img_src = os.path.join(model_image_folder, img_file)
    
    # Return the relative path to the image (as the 'assets' folder is served by Dash)
    return img_src
