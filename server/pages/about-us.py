import dash
from dash import html, dcc, callback, Input, Output
from components.profile import Profile
import dash_bootstrap_components as dbc

dash.register_page(__name__)

styles = {"padding": "20px"}
description_placeholder = """
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
image_placeholder = "https://via.placeholder.com/200"
bespoke_robot_image = dbc.Col(
    [
        html.Img(
            src="/assets/brs_transparency.png",  # replace with your image URL
            alt="Profile Picture",
            style={"width": "300px", "height": "300px", "padding": "0px"},
        ),
    ],
    width={"size": 2, "offset": 5},
    className="text-center",
)
andy_profile = Profile("Andy Ponce", description_placeholder, "/assets/AndyPhoto.JPG")
profile_placeholder = Profile("Your Name", description_placeholder, image_placeholder)

layout = html.Div(
    [
        html.H1("About Us"),
        bespoke_robot_image,
        profile_placeholder,
        andy_profile,
        profile_placeholder,
        profile_placeholder,
        profile_placeholder,
        profile_placeholder,
    ],
    style=styles,
)
