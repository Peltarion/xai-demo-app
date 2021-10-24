#helper functions – helper.py file
import shap
import pickle
import pandas as pd

# transform categorical data
def convert_to_series(text):
    """
    When you get the raw data, it needs to be processed and make it ready for scoring
    """
    print("I am in convert to series")
    print(text)
    return pd.Series(text)

# score new data
def score_record(input_text, predictor):
    result = predictor(input_text)[0]
    keys = []
    values = []
    for d in result:
        pair = list(d.values())
        keys.append(pair[0])
        values.append(pair[1])
    dict_result = dict((k, v) for k, v in zip(keys, values))
    predicted_class = max(dict_result, key=dict_result.get)
    prob = max(dict_result.values())
    return predicted_class, prob
