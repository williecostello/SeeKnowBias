import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate


# Data science packages
import pandas as pd
import joblib
import logging
import os
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import nltk
# from nltk.corpus import stopwords
# english_stop_words = stopwords.words('english')
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

from assets.exs import nyt, foxnews, post, jacobin, breitbart, newsweek
from dl_model import download


download()

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css"
    ],
)

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

def load_model(file_path):
    pickle = open(file_path, 'rb')
    model = joblib.load(pickle)
    pickle.close()
    return model


model = load_model("./model.pkl")


@app.callback([Output('textarea', 'value'),
                Output("div", "children")],
            [Input("breitbart", "n_clicks"),
            Input("fox-news", "n_clicks"),
            Input("jacobin", "n_clicks"),
            Input("new-york-times", "n_clicks"),
            Input("new-york-post", "n_clicks"),
            Input("newsweek", "n_clicks"),
            Input("analyze-btn", "n_clicks")],
            [State("textarea", "value")])
def show_premade(breit_btn,
                 fox_btn,
                 jaco_btn,
                 nyt_btn,
                 nyp_btn,
                 newsweek_btn,
                 analyze_btn,
                 textarea):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id:
        if button_id=="breitbart":
            return breitbart, breitbart
        if button_id=="fox-news":
            return foxnews, foxnews
        if button_id=="jacobin":
            return jacobin, jacobin
        if button_id=="new-york-times":
            return nyt, nyt
        if button_id=="new-york-post":
            return post, post
        if button_id=="newsweek":
            return newsweek, newsweek
        if button_id=="analyze-btn":
            return textarea, textarea
    return "", ""


@app.callback(Output("output", "children"),
            [Input("div", "children"),
            Input("confidence-slider", "value"),
            Input("num-words-slider", "value")])
def process_text(text, thresh, num_biased):
    """
    Take text, process, and output
    """
    if text is None or len(text)==0:
        raise PreventUpdate

    sentences = pd.Series(sent_tokenize(text))
    bias_probas = model.predict_proba(sentences)

    children = [html.Div("Hover over the word to see exact score", className='text-center font-italic mb-2')]
    currentP_list = []
    check = False
    #### use title on spans for hover value
    for i, art in enumerate(sentences):
        if i%3==0 or check:
            
            if len(art.strip()) == 0:
                check = True
            else:
                logging.critical(f"i={i} {art}")
                children.append(html.P(" ".join(currentP_list), className="para"))
                children.append(html.P(" ", className="break"))  
                currentP_list = []
                check = False
        if bias_probas[i][1] > thresh:
            children.append(html.P(" ".join(currentP_list), className="para"))
            currentP_list = []
            words = art.split(' ')
            new_sentences = pd.Series()
            for j in range(len(words)):
                new_sentence = words.copy()
                del new_sentence[j] # remove the one word
                new_sentence = ' '.join(new_sentence)
                new_row = pd.Series(new_sentence)
                new_sentences = new_sentences.append(new_row, ignore_index=True)
                sent_probas = model.predict_proba(new_sentences)
            
            sent_ids = pd.DataFrame(sent_probas)[1].nsmallest(num_biased).index.values
            for k, word in enumerate(words):
                
                if k in sent_ids:  
                    children.append(html.P(" "+" ".join(currentP_list)+" ", className="para"))
                    currentP_list = []
                    children.append(html.P(" "))
                    children.append(get_bias_html(word, bias_probas[i][1]))
                    children.append(html.P(" "))
                                    
                else:
                    currentP_list.append(word)
        else:
            if len(currentP_list)==0:
                currentP_list.append(" "+art)
            else:
                currentP_list.append(art)

    if len(currentP_list) > 0:
        children.append(html.P(" ".join(currentP_list)))

    return children

def get_bias_html(word, proba):

    return html.Span(
                children=word,
                title=f"Bias probability = {int(proba*100)}%",
                className=f"biased bias-{int(proba*10)} rounded p-1",
            )
    

def isclose(a, b, rel_tol=1e-04, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8051)

