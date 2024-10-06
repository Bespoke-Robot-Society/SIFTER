import dash
from dash import html, dcc, callback, Input, Output
from components.profile import Profile
import dash_bootstrap_components as dbc
import os

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

def bio_text(fn):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "..", fn), "r") as f:
        return f.read()

aaron_profile = Profile("Aaron W. Storey", bio_text("assets/aaron.txt"), "/assets/AaronPhoto.JPG")
andy_profile = Profile("Andy Ponce", bio_text("assets/andy.txt"), "assets/AndyPhoto.JPG")
john_profile = Profile("John P. McCardle III", bio_text("assets/john.txt"), "/assets/JohnPhoto.JPG")
jason_profile = Profile("Jason Jain", bio_text("assets/jason.txt"), image_placeholder)
#lee_profile
robert_profile = Profile("Robert Turner", bio_text("assets/rob.txt"), image_placeholder)
profile_placeholder = Profile("Your Name", description_placeholder, image_placeholder)

layout = html.Div(
    [
        html.H1("About Us"),
        bespoke_robot_image,
        aaron_profile,
        jason_profile,
        andy_profile,
        robert_profile,
        profile_placeholder,
        john_profile,
    ],
    style=styles,
)
