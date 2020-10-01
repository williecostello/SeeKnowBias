import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate

import pandas as pd

from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import joblib
import logging

from assets.exs import nyt, foxnews, post, jacobin, breitbart, newsweek


app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css"
    ],
)


server = app.server


news_pairs = [
("Breitbart", "breitbart"),
("Fox News", 'fox-news'),
("New York Times", "new-york-times"),
("New York Post", "new-york-post"), 
("Jacobin", "jacobin"),
("Newsweek",  "newsweek") 
]


def get_news_btn(news_label, news_id):
    return html.Button(
                    news_label,
                    id=news_id,
                    className="btn btn-dark mx-2 p-2 col rounded text-center",
                    n_clicks=0,
                )


app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(children="See.Know.Bias", className="display-1"),
                # html.H3("Copy & paste an article into the text box below"),
                html.Div(children=[
                    dcc.Textarea(id="textarea", 
                            className="col-10 rounded border border-dark", 
                            value="",
                            placeholder="Copy & paste an article into the text box, then hit \"Run\"",
                            autoFocus="true"),
                    html.Button("Run", id="analyze-btn", className="col-2 btn btn-dark text-center "),
                ],
                className="row"),
                
                html.H3("Or select a pre-loaded article from one of the news sources below", className='mt-3'),
                html.H4("(All articles concern Alexandria Ocasio-Cortez's 2020 primary election win)",
                    className="font-italic font-weight-lighter mb-4"),
                html.Div(children=[get_news_btn(a[0], a[1]) for a in news_pairs],
                className="row"),
                html.H5(
                    "Set bias confidence threshold (how confident the algorithm should be that a sentence is biased)",
                    className="mt-5"),
                dcc.Slider(
                    id="confidence-slider",
                    min=0.5,
                    max=1.0,
                    step=0.05,
                    value=0.7,
                    marks={ i/100.0: str(i/100.0) for i in range(50,121,5)},
                    dots=True,
                    className="mt-3, mb-4"
                ),
                html.H5("Set number of biased words to highlight in each sentence"),
                dcc.Slider(id="num-words-slider",
                            min=1,
                            max=3,
                            step=1,
                            value=1,
                            dots=True,
                            marks={i: str(i) for i in range(1,4)},
                            className="my-3"),
                html.Div(id="div", style={'display':'none'}),
                html.Div(children=[
                html.P("50-59%", className='bias-5 m-4 p-2 col rounded text-center'),
                html.P("60-69%", className='bias-6 m-4 p-2 col rounded text-center'),
                html.P("70-79%", className='bias-7 m-4 p-2 col rounded text-center'),
                html.P("80-89%", className='bias-8 m-4 p-2 col rounded text-center'),
                html.P("90-100%", className='bias-9 m-4 p-2 col rounded text-center'),
                ],
                className="row"),
                html.Div(id='output-wrapper', className="pb-5 d-flex justify-content-center",
                children=html.Div(id='output', className='px-5'))
             ]),
    ],
    className="container")




if __name__ == "__main__":
    app.run_server()