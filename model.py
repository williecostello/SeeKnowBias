import pandas as pd

from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import joblib

def process_data(file_path):
    
    # Read in training data
    df = pd.read_csv(file_path, sep='\t', names=['id', 'src_tok', 'tht_tok', 'src_raw', 'tgt_raw', 'src_POS', 'tgt_POS'])
    
    # Create new dataframe from just raw biased & corrected sentences
    bias_sents = df['src_raw']
    neut_sents = df['tgt_raw']

    bias_df = pd.DataFrame({'sentence':bias_sents.values, 'biased':1})
    neut_df = pd.DataFrame({'sentence':neut_sents.values, 'biased':0})
    sent_df = bias_df.append(neut_df).reset_index(drop=True)
    
    return sent_df

data_path = '~/Downloads/bias_data/WNC/'

df_train = process_data(data_path + 'biased.word.train')
df_dev = process_data(data_path + 'biased.word.dev')
df_test = process_data(data_path + 'biased.word.test')

df_full = df_train.append(df_dev).append(df_test).reset_index(drop=True)
X = df_full['sentence']
y = df_full['biased']

model = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2))), 
                  ('log', LogisticRegression(C=1)),])

model = model.fit(X, y);

joblib.dump(model, 'model.pkl');