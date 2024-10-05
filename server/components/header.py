import dash
from dash import Dash, html, dcc

# Define CSS styles for the header
styles = {
    "width": "100%",
    "display": "flex",
    "flex-grow": "1",
    "border-bottom": "1px solid #ccc",  # Add a border at the top
    "padding": "20px",  # Add some padding for spacing
    "background-color": "#f2f2f2",  # Set a background color
}

links = [{"label": "Home", "href": "/"}, {"label": "About Us", "href": "/about-us"}]


def get_header():
    header = html.Header(
        children=[
            html.H2("S.I.F.T.E.R.", style={"flex": 1}),
            html.Div(
                [
                    html.Div(
                        dcc.Link(
                            link["label"],
                            href=link["href"],
                        ),
                        style={"margin": "0 10px"},
                    )
                    for link in links
                ],
                style={
                    "display": "flex",
                    "align-items": "center",
                },
            ),
        ],
        style=styles,
    )
    return header
