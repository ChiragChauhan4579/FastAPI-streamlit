import json
import math
import os
import requests
os.environ['NO_PROXY'] = '127.0.0.1'
import streamlit as st

st.title("BentoML Streamlit Demo")

sl = st.number_input('Enter value for sepal length')
sw = st.number_input('Enter value for sepal width')
pl = st.number_input('Enter value for petal length')
pw = st.number_input('Enter value for petal width')

data = [sl,sw,pl,pw]

if st.button("Predict"):
    if not any(math.isnan(v) for v in data):

        prediction = requests.post(
            "https://fastapi-test-f74q.onrender.com/predict",
            headers={"content-type": "application/json"},
            json={"sepal_length":sl,
            "sepal_width":sw,
            "petal_length":pl,
            "petal_width":pw}
        ).json() #text for returning simple text

        st.write(f"This flower belongs to class {prediction['class']}")
