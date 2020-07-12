import urllib.request
from pathlib import Path

import logging

logging.critical('Beginning file download with urllib2...')

url = 'https://docs.google.com/uc?export=download&id=1x__YR6PO6_cW-vy3vb0cdjWRYqjBqPW0'
urllib.request.urlretrieve(url, Path(__file__).parent.resolve()/"model.pkl")

logging.critical('Finished download')