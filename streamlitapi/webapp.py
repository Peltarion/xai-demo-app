import streamlit as st
import requests
import datetime
import shap
import json
import pickle
import pandas as pd
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from PIL import Image
import time

st.set_page_config(page_title='Peltarion XAI Demo', page_icon = 'https://connectoricons-prod.azureedge.net/releases/v1.0.1410/1.0.1410.2197/peltarion/icon.png', layout = 'wide', initial_sidebar_state = 'auto')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.image('https://linkopingsciencepark.se/wp-content/uploads/2020/09/peltarion-logotype-vertical-red.png', width=150,  output_format='JPEG')
#st.sidebar.image('https://connectoricons-prod.azureedge.net/releases/v1.0.1410/1.0.1410.2197/peltarion/icon.png', output_format='JPEG')
st.title('Real-time Explainable Predictions')
#image = Image.open('./logo.png')
#st.sidebar.image(image, caption='')

# Python API endpoint
url = 'http://pythonapi:5000'
endpoint = '/api/'
st.sidebar.header('Insert text here:')

def user_input_features():
    input_features = {}
    
    input_features["user_text"] = st.sidebar.text_area("","This movie makes life look like anything but an endless sadness.")
    
    input_features["emotion"] = st.sidebar.selectbox("select a class to explain:", ["sadness", "joy", "love", "anger", "fear", "surprise"])
    return input_features

json_data = user_input_features()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

with open('score_objects.pkl', 'rb') as handle:
    predictor, explainer = pickle.load(handle)

def explain_model_prediction(data):
    shap_values = explainer(data) 
    p=shap.plots.bar(shap_values[:, :, "joy"])
    return p, shap_values

submit = st.sidebar.button('Get predictions')
explain = st.sidebar.button('Explain')
results = requests.post(url+endpoint, json=json_data)
results = json.loads(results.text)
results = pd.DataFrame([results])
emotion = results.loc[0, "emotion"]
prediction = results.loc[0,"prediction"]
probability = results.loc[0,"probability"]
text = results.loc[0,"user_text"]
if submit:
    st.header('Prediction:')
    st.write("Predicted class: ", prediction)
    st.write("Probability: ", round(float(probability),3))
if explain:
    st.header('Prediction:')
    st.write("Predicted class: ", prediction)
    st.write("Probability: ", round(float(probability),3))
    #plots 
    st.header('Interpretation Plot:')
    st.write("Selected class to be explained: ", emotion)
    with st.spinner('Please Wait...'):
        shap_values = explainer(pd.Series(text))
    st.success('Done!')
    #st.balloons()
    st.write(shap.plots.bar(shap_values[:, :, emotion].mean(0)), unsafe_allow_html=True)
    st.pyplot(bbox_inches='tight')
