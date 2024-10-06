import dash
from dash import html

dash.register_page(__name__, path="/")

styles = {"padding": "20px"}


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
    ],
    style=styles,
)
