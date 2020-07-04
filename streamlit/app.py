# Web app packages
import streamlit as st

# Data science packages
import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import nltk
# from nltk.corpus import stopwords
# english_stop_words = stopwords.words('english')
from nltk.tokenize import sent_tokenize

from exs import nyt, foxnews, post, jacobin, breitbart, newsweek

'''
# See.Know.Bias
'''

@st.cache(allow_output_mutation=True)
def load_model(file_path):
    pickle = open(file_path, 'rb')
    model = joblib.load(pickle)
    return model

model = load_model('model.pkl')

st.sidebar.markdown(
    '''
    ### Copy & paste an article into the text box on the right

    ### Or select a pre-loaded article from one of the news sources below
    
    (*All articles concern Alexandria Ocasio-Cortez's 2020 primary election win*)
    '''
)

example = ''
if st.sidebar.button('Breitbart'):
    example = breitbart
if st.sidebar.button('Fox News'):
    example = foxnews
if st.sidebar.button('Jacobin'):
    example = jacobin
if st.sidebar.button('New York Times'):
    example = nyt
if st.sidebar.button('New York Post'):
    example = post
if st.sidebar.button('Newsweek'):
    example = newsweek

text = st.text_area('Copy & paste an article into the text box below', value=example, height=300)

thresh = st.slider('Set bias confidence threshold (how confident the algorithm should be that a sentence is biased)', min_value=.5, max_value=.9, value=.7, step=.05)

num_biased = st.slider('Set number of biased words to highlight in each sentence', min_value=1, max_value=3, value=1)
 
'Neutral sentences are displayed in *italics*. Biased sentences are offset, with the most biased words highlighted in **bold**.'
'---'

if text != '':

    sentences = sent_tokenize(text)
    article = pd.Series(sentences)
    bias_check = model.predict(article)
    bias_probas = model.predict_proba(article)

    for i in range(len(article)):
        if bias_probas[i][1] > thresh:
            words = article[i].split(' ')
            new_sentences = pd.Series()
            for j in range(len(words)):
                new_sentence = words.copy()
                del new_sentence[j]
                new_sentence = ' '.join(new_sentence)
                new_row = pd.Series(new_sentence)
                new_sentences = new_sentences.append(new_row, ignore_index=True)
            sent_probas = model.predict_proba(new_sentences)
            sent_ids = pd.DataFrame(sent_probas)[1].nsmallest(num_biased).index.values
            for k in range(len(words)):
                if k in sent_ids:
                    words[k] = '**' + words[k] + '**'
            biased_sent = ' '.join(words)
            f'> **Biased ({int(round(bias_probas[i][1], 2)*100)}% confidence):** {biased_sent}'
        else:
            f'*{article[i]}*'