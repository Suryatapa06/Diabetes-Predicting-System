
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:40:09 2025

@author: gopas
"""
import os
import numpy as np
import pickle
import streamlit as st

# Print current working directory in Streamlit app
st.write("Current Working Directory:", os.getcwd())

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

#creating a function for prediction

def diabetes_prediction(input_data):
  
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
def main():
    # Title of the page
    st.title("Diabetes Prediction Web App")

    # Inputs from the user
    Pregnancies = st.text_input('Number of pregnancies (e.g., 2): ')
    Glucose = st.text_input('Glucose Level (e.g., 85): ')
    BloodPressure = st.text_input('Blood Pressure Level (e.g., 66): ')
    SkinThickness = st.text_input('Skin Thickness (e.g., 29): ')
    Insulin = st.text_input('Insulin Level (e.g., 94): ')
    BMI = st.text_input('BMI Level (e.g., 26.6): ')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function (e.g., 0.351): ')
    Age = st.text_input('Age (e.g., 31): ')

    # Code for prediction
    diagnosis = ''

    # Validate and convert inputs
    if st.button('TEST'):
        try:
            # Convert inputs to float, replacing empty inputs with 0 or a default value
            input_data = [
                float(Pregnancies) if Pregnancies else 0.0,
                float(Glucose) if Glucose else 0.0,
                float(BloodPressure) if BloodPressure else 0.0,
                float(SkinThickness) if SkinThickness else 0.0,
                float(Insulin) if Insulin else 0.0,
                float(BMI) if BMI else 0.0,
                float(DiabetesPedigreeFunction) if DiabetesPedigreeFunction else 0.0,
                float(Age) if Age else 0.0
            ]
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = "Invalid input. Please ensure all fields are numeric."

        st.success(diagnosis)


if __name__ == '__main__':
    main()