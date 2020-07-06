import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css"
    ],
)

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(children="See.Know.Bias"),
                html.H3("Copy & paste an article into the text box below"),
                dcc.Textarea(id="textarea", className="w-100"),
                html.H3("Or select a pre-loaded article from one of the news sources below"),
                html.H4("(All articles concern Alexandria Ocasio-Cortez's 2020 primary election win)"),
                html.Button(
                    "Breitbart", id="breitbart", className="btn btn-light mx-2", n_clicks=0
                ),
                html.Button(
                    "Fox News", id="fox-news", className="btn btn-light mx-2", n_clicks=0
                ),
                html.Button(
                    "Jacobin", id="jacobin", className="btn btn-light mx-2", n_clicks=0
                ),
                html.Button(
                    "New York Times",
                    id="new-york-times",
                    className="btn btn-light mx-2",
                    n_clicks=0,
                ),
                html.Button(
                    "New York Post",
                    id="new-york-post",
                    className="btn btn-light mx-2",
                    n_clicks=0,
                ),
                html.Button(
                    "Newsweek", id="newsweek", className="btn btn-light mx-2", n_clicks=0
                ),
                html.H5(
                    "Set bias confidence threshold (how confident the algorithm should be that a sentence is biased)"
                ),
                dcc.Slider(
                    id="confidence-slider", min=0.5, max=1.0, step=0.05, value=0.7,
                ),
                html.H5("Set number of biased words to highlight in each sentence"),
                dcc.Slider(id="num-words-slider", min=1, max=3, step=1, value=1),
                html.H5(
                    "Neutral sentences are displayed in italics. Biased sentences are offset, with the most biased words highlighted in bold."
                ),
                html.Hr(),
                
            ]
        ),
    ],
    className="container")

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)

