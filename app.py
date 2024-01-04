# Importing necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from page_contents import business_understanding, load_data, data_understanding, data_preparation, modeling, evaluation, prediction
import pickle

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Setting up the Streamlit page
st.set_page_config('Titanic Dataset Analysis and Modeling', layout='wide')

# Sidebar content
with st.sidebar:
    st.header('Titanic Analysis and Modeling')
    st.image('logo.png')
    st.write('---')
    step = st.radio(
        'Choose the step you want to see: ',
        ['1. Business Understanding', '2. Data Understanding', '3. Data Preparation', '4. Modeling', '5. Evaluation', '6. Predict Yourself']
    )

# Main page content
df = load_data()
df2 = df.copy()

if step == '1. Business Understanding':
     business_understanding()
elif step == '2. Data Understanding':
    data_understanding(df2)
elif step == '3. Data Preparation':
    data_preparation()
elif step == '4. Modeling':
    modeling()
elif step == '5. Evaluation':
    evaluation()
elif step == '6. Predict Yourself':
    prediction(scaler)
