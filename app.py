""" 

Application that predicts heart disease percentage in the population of a town based on the number of bikers and  smokers.

Trained on the data set of percentage of people biking to work each day, the percentage of people who smoke, and the percentage of people who have heart disease in an imaginary town.

"""

# Importing the libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Creating the Flask app
app = Flask(__name__)

# Loading the model
model = pickle.load(open('models/model.pkl', 'rb'))

# Define the route to be home
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to be prediction
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Get the features
    features = [float(x) for x in request.form.values()]
    # Convert the features to a numpy array
    final_features = [np.array(features)] # Convert to th form [[x1, x2]]
    # Reshape the array
    features_df = pd.DataFrame(final_features, columns=['biking', 'smoking'])
    # Make prediction
    prediction = model.predict(features_df)
    # Get the output
    output = round(prediction[0], 2)
    # Render the result to the HTML GUI
    return render_template('index.html', prediction_text='Percentage of people with heart disease: {}'.format(output))


if __name__ == "__main__":
    app.run(host='localhost', port=8000)