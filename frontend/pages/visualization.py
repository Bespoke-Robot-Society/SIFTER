import os
import dash
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

# Register Dash page
dash.register_page(__name__, path="/")

# Define the path to the folder containing the images
image_folder = 'frontend/assets/images/output_graphs'
rel_image_folder = 'assets/images/output_graphs'

# Get a list of all PNG files in the folder
image_list = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# Layout
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
            html.Label("Select an Image to View", className="h5"),
            dcc.Dropdown(
                id='image-dropdown',
                options=[{'label': img, 'value': img} for img in image_list],
                value=image_list[0] if image_list else None,
                clearable=False
            ),
        ], width=6, className='mx-auto')
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Img(id='image-display', style={'width': '100%', 'height': 'auto'}),
            html.P(id='dynamic-text', className="mt-3")
        ], width=10, className='mx-auto')
    ], className="mb-5"),

    # About Section
    dbc.Row([
        dbc.Col([
            html.H2("About Our Model", className="my-4"),
            html.P(
                """The S.I.F.T.E.R. project utilizes a hybrid approach combining two models: STA/LTA (Short-Term Average/Long-Term Average) method and a Convolutional Neural-Network (CNN) model to analyze seismic data. The STA/LTA trigger identifies potential seismic event arrivals by detecting significant changes in the signal's amplitude over short and long time windows. When the STA/LTA ratio exceeds a threshold of 3.5, it flags relevant time windows where an event might have occurred, which then passes the interesting time periods to produce a spectrogram for the CNN to analyze."""
            ),
            html.P(
                """Generated spectrograms are then passed to our CNN, which is trained to identify seismic events and predict the precise event arrival time. CNNs are commonly used in image analysis, because they are able to analyze neighboring blocks (like pixels) in their input. We are passing our CNN an image of a spectrogram, so features like sharp edges in time and frequency become features the model can detect. The CNN model is designed to handle potentially noisy seismic data with the use of multiple convolutional layers, batch normalization, and dropout for regularization. The convolutional layers help the model extract local patterns in the spectrograms, such as characteristic frequency bursts associated with seismic events, while max-pooling reduces dimensionality and focuses on the most important features. The CNN refines the estimates provided by STA/LTA by analyzing deeper patterns in the spectrogram."""
            )
        ], width=10)
    ], className="justify-content-center")
], fluid=True)

# Callback to update image and text
@callback(
    [Output('image-display', 'src'),
     Output('dynamic-text', 'children')],
    [Input('image-dropdown', 'value')]
)
def update_image(value):
    # Update image source
    image_src = f'/{rel_image_folder}/{value}'
    
    # Update text description based on selected image
    text = f"This image represents seismic data for {value}."
    
    return image_src, text
