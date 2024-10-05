import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(__name__)

styles = {"padding": "20px"}

layout = html.Div(
    [
        html.H1("About Us"),
    ],
    style=styles,
)
