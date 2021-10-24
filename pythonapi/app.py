from flask import Flask, request, redirect, url_for, flash, jsonify, make_response
import pandas as pd
import numpy as np
import pickle
import json
import shap
from helper import *

with open('score_objects.pkl', 'rb') as handle:
    predictor, explainer = pickle.load(handle)

app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def makecalc():
    """
    We get the data from the Streamlit front end API in json format. 
    Then, we use the helper functions to perform preprocessing and produce the scores. 
    Finally, we sent back the scores to Streamlit front end API.
    """
    json_data = request.get_json()
    # json_data will be like this: {"user_text": "This was an awesome movie!"}
    # that is given by your API (user gave it to API)

    #read the real time input to pandas df
    data = pd.DataFrame([json_data])
    print(data)
    
    #transform DataFrame
    text = data.loc[0,'user_text']
    
    #score df
    prediction, probability = score_record(text, predictor)
    
    #convert predictions to dictionary
    data['prediction'] = prediction
    data['probability'] = probability
    output = data.to_dict(orient='rows')[0]
    
    #output['plot'] = p
    return jsonify(output)

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)
