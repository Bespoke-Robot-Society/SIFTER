import dash
from dash import Dash, html
import dash_bootstrap_components as dbc
from components.header import get_header
from components.footer import get_footer

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    use_pages=True,
)

app.layout = html.Div(
    [
        get_header(),
        dash.page_container,
        get_footer(),
    ]
)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
