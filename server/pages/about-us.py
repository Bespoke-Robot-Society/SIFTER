import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
dash.register_page(__name__)

styles = {"padding": "20px"}
profile_desc_ex = \
"""
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam rhoncus tempus vestibulum. 
Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Maecenas ante felis, sagittis ac convallis non, congue sit amet libero. Fusce ac accumsan augue. 
Praesent quis ante est. Duis tristique porta magna, non mattis quam efficitur in. 
Ut ac orci sed elit condimentum cursus. Fusce consequat purus in tincidunt volutpat. 
Duis iaculis, odio vel pharetra maximus, ipsum neque rhoncus sapien, eget dictum purus eros a nibh. 
Nunc nibh libero, iaculis sit amet vehicula in, auctor nec ante. Mauris imperdiet sodales quam, at lacinia ante efficitur sed. 
Curabitur malesuada pellentesque rutrum. 
Proin auctor ligula sem, et tristique magna iaculis mollis. Nulla ac dui justo. 
"""
single_profile_ex = \
                dbc.Row(
            [
                dbc.Col(
                    html.Img(
                        src="https://via.placeholder.com/200",  # replace with your image URL
                        alt="Profile Picture",
                        className="img-fluid rounded-circle",
                        style={"width": "200px", "height": "200px"}
                    ),
                    width={"size": 2, "offset": 1},
                    className="text-center"
                ),
                dbc.Col(
                    [
                        html.H3("Your Name"),
                        html.P(profile_desc_ex),
                        html.P("Based in: City, Country"),
                    ],
                    width=7
                ),
            ],
            className="mt-4"
        )
layout = html.Div(
    [
        html.H1("About Us"),
        single_profile_ex,
        single_profile_ex,
        single_profile_ex,
        single_profile_ex,
        single_profile_ex,
        single_profile_ex,
    ],
    style=styles,
)
