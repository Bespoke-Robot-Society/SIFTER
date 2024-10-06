from dash import html
import dash_bootstrap_components as dbc

class Profile:
    def __new__(cls, name, description, image_path, image_width=200, image_height=200):
        instance = super().__new__(cls)
        instance.__init__(name, description, image_path, image_width, image_height)
        # Return the profile directly instead of the instance
        return instance.profile
    def __init__(self, name, description, image_path, image_width = 200, image_height = 200):
        """
        Create a Dash component representing a single profile.

        Parameters
        ----------
        name : str
            The name of the person.
        description : str
            A short description of the person.
        image_path : str
            The path to the image for the profile (usually in the assets folder).
        image_width : int, optional
            The width of the image in pixels, defaults to 200px.
        image_height : int, optional
            The height of the image in pixels, defaults to 200px.

        Returns
        -------
        dash_bootstrap_components.Row
            A Dash boostrap Row component representing the profile.
        """
        self.name = name
        self.description = description
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height
        self.profile = \
            dbc.Row(
            [
                dbc.Col(
                    html.Img(
                        src=self.image_path,
                        alt=f"Picture of {self.name}",
                        className="img-fluid rounded-circle",
                        style={"width": f"{self.image_width}px", "height": f"{self.image_height}px"}
                    ),
                    width={"size": 2, "offset": 1},
                    className="text-center"
                ),
                dbc.Col(
                [html.H3(self.name)] + [html.P(txt) for txt in self.description.split('\n\n')],
                width=7
            )
            ],
            className="mt-4"
        )
