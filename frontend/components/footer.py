import dash
from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc

# Disclaimer footer component
def get_footer():
    footer = dbc.Container(
        dbc.Row(
            dbc.Col(
                html.P(
                    [
                        html.Strong("Disclaimer: "),
                        "This version of the project was submitted on 10/6/2024 for the 2024 NASA Space Apps Challenge. No further updates to this project or code will be posted to this site until judging has been finalized. For the most recent updated version of the project, please visit the",
                        html.A("project repo on GitHub", href="https://github.com/Bespoke-Robot-Society/SIFTER"),
                        "."
                    ],
                    className="text-center"
                ),
                width=12
            ),
            className="my-2"
        ),
        fluid=True,
        className="bg-dark text-light py-3"
    )
    
    return footer