import dash
from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc

# Navigation links
links = [{"label": "Home", "href": "/"}, {"label": "About Us", "href": "/about-us"}]

def get_header():
    navbar = dbc.Navbar(
        dbc.Container(
            [
                # Logo and Title
                dbc.Row(
                    [
                        dbc.Col(
                            html.A(
                                html.Img(
                                    src='/assets/images/sifter_cicdbot_icon.png',
                                    alt="S.I.F.T.E.R. Logo",
                                    height="50px",
                                    className="rounded-circle",
                                ),
                                href="/",
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.NavbarBrand("S.I.F.T.E.R.", className="ml-2"),
                            width="auto",
                        ),
                    ],
                    align="center",
                    className="g-0",
                ),
                # Toggle button for mobile view
                dbc.NavbarToggler(id="navbar-toggler"),
                # Navigation Links
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink(link["label"], href=link["href"]))
                            for link in links
                        ],
                        className="ml-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ],
            fluid=True,
        ),
        color="dark",
        dark=True,
        sticky="top",
    )
    return navbar

# Callback to toggle the collapse on small screens
@callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open
