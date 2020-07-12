import urllib.request
from pathlib import Path

import logging



def download():
    if not (Path.cwd()/"model.pkl").exists()
        url = 'https://docs.google.com/uc?export=download&id=1x__YR6PO6_cW-vy3vb0cdjWRYqjBqPW0'
        urllib.request.urlretrieve(url, "./model.pkl")