import urllib.request
from pathlib import Path

import logging

def download():
    if (Path.cwd()/"model.pkl").exists() is False:
        url = 'https://drive.google.com/uc?export=download&id=1hjRGMVQYAIHCx5J8D9Pd64Ut2jOvQDgh'
        urllib.request.urlretrieve(url, "./model.pkl")