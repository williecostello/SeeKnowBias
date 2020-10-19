# Download & unzip data
echo 'Downloading data'
curl https://nlp.stanford.edu/projects/bias/bias_data.zip -o bias_data.zip
echo 'Data downloaded!'

echo 'Unzipping data'
unzip bias_data.zip -d data
echo 'Data unzipped!'

# Build & pickle model
'Building model'
python model.py