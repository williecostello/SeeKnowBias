## sklearn model

from flask import Flask, request, jsonify, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, validators
import mlflow.pyfunc
import pandas as pd
import json

# Name of the apps module package
app = Flask(__name__)

app.config['SECRET_KEY'] = 'key'

# Load in the model at app startup
model = mlflow.pyfunc.load_model('./model')

# Load in our meta_data
with open("./meta_data.txt", "r") as f:
	load_meta_data = json.loads(f.read())



class MyForm(FlaskForm):
	textarea = StringField("Input_text", [validators.DataRequired()],
		render_kw={"placeholder":"Input text here"})	

# Meta data endpoint
@app.route('/', methods=['GET', 'POST'])
def index():
	form = MyForm()

	if request.method == "GET":
		result = None
	if request.method == "POST":
		text = form.textarea.data

		# Format the request data in a DataFrame
		# payload example:
		# {
		# 'text': ['first sent', 'second']
		# }
		inf_df = pd.DataFrame(data={'text': [text]})

		# Get model prediction - convert from np to list
		result = model.predict(inf_df).tolist()



	return render_template('index.html', form=form, result=result, meta=load_meta_data)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
	req = request.get_json()
	
	# Log the request
	print({'request': req})

	# Format the request data in a DataFrame
	# payload example:
	# {
	# 'text': ['first sent', 'second']
	# }
	inf_df = pd.DataFrame(data={'text': req['data']})

	# Get model prediction - convert from np to list
	pred = model.predict(inf_df).tolist()

	# Log the prediction
	print({'response': pred})

	# Return prediction as reponse
	return jsonify(pred)

app.run(host='0.0.0.0', port=5000, debug=True)
