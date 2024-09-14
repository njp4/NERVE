# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:42:47 2024

@author: nicho
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('random_forest_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Function to predict nerve deficit with standard deviation
def predict_nerve_deficit_with_std(age, hypertension, mixed_nerve, days_from_injury):
    input_data = pd.DataFrame({
        'age': [age],
        'hypertension': [hypertension],
        'mixed_nerve': [mixed_nerve],
        'days_from_injury': [days_from_injury]
    })

    # Get predictions from all individual trees in the forest
    tree_predictions = np.array([tree.predict(input_data) for tree in rf_model.estimators_])
    
    # Calculate the mean prediction
    predicted_deficit = np.mean(tree_predictions)
    
    # Calculate the standard deviation of the predictions
    std_dev = np.std(tree_predictions)
    
    return predicted_deficit, std_dev

# Streamlit app title
st.title('Nerve Deficit Prediction Calculator')

# Input form for user
age = st.number_input('Age', min_value=0, max_value=120, value=30)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
mixed_nerve = st.selectbox('Mixed Nerve', ['No', 'Yes'])
days_from_injury = st.number_input('Days from Injury', min_value=0, max_value=365, value=10)

# Convert user inputs to model-friendly format
hypertension = 1 if hypertension == 'Yes' else 0
mixed_nerve = 1 if mixed_nerve == 'Yes' else 0

# Predict button
if st.button('Predict'):
    predicted_deficit, std_dev = predict_nerve_deficit_with_std(
        age, hypertension, mixed_nerve, days_from_injury)
    st.write(f'Predicted Nerve Deficit: {predicted_deficit:.2f} mm')
    st.write(f'Standard Deviation: {std_dev:.2f} mm')
    
#nerve evaluation and retraction variability estimator