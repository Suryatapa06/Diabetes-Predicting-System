
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
    
    
    #title of the page
    st.title("Diabetes Prediction Web App")
    
    #inputs from the user
    #inputs to be taken : Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    Pregnancies = st.text_input('Number of pregnancies: ')
    Glucose = st.text_input('Glucose Level: ')
    BloodPressure = st.text_input('Blood Pressure Level: ')
    SkinThickness = st.text_input('Skin Thickness: ')
    Insulin = st.text_input('Insulin Level: ')
    BMI = st.text_input('BMI Level: ')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function: ')
    Age = st.text_input('Age: ')
    
    
    #code for prediction
    
    diagnosis = ''
    
    #creating button for testing the result
    
    if st.button('TEST'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        st.success(diagnosis)
    
    
    
    
if __name__ == '__main__':
    main()