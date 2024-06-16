# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:43:23 2024

@author: dell
"""

import numpy as np 
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))
loaded_scaler = pickle.load(open('scaler_file.sav','rb'))
#creating a function for prediction

def breast_cancer_prediction(input_data):
    
    #change the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the numpy array as we are predicting for one data point
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    #standardizing the input data
	input_data_std = loaded_scaler.transform(input_data_reshaped)

	prediction = loaded_model.predict(input_data_std)
	print(prediction)

	prediction_label = [np.argmax(prediction)]
	print(prediction_label)

	if(prediction_label[0]==0):
  	print('The tumor is Malignant')
	else:
 	print('The tumor is Benign')
  
    
def main():
    
    #giving a title
    st.title('Breast Cancer Prediction Web App')
    
    #taking input from the user
   #test_values = (st.text_input("Input whole test value"))
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        mean_radius = st.text_input('mean radius (5-30):')

    with col2:
        mean_texture  = st.text_input('mean texture(tens) ')
        
    with col3:
        mean_perimeter  = st.text_input('mean perimeter(<>=100)')
                                      
    with col4:
        mean_area  = st.text_input('mean area (<>=1000)')

    with col5:
        mean_smoothness = st.text_input('mean smoothness(0.)')
        
    #second
    
        
    with col1:
        mean_compactness = st.text_input('mean_compactness(0.):')

    with col2:
        mean_concavity = st.text_input('mean concaivity(0.) ')
        
    with col3:
        mean_concave_points = st.text_input('mean concave points(0.)')
                                      
    with col4:
        mean_symmetry = st.text_input('mean symmetry(0.) ')

    with col5:
        mean_fractal_dimension = st.text_input('mean fractal dimension(0.)')
        
    #Third
    
    with col1:
        radius_error = st.text_input('radius error:')

    with col2:
        texture_error = st.text_input('texture error')
        
    with col3:
        perimeter_error  = st.text_input('perimeter error')
                                      
    with col4:
        area_error = st.text_input('area error ')

    with col5:
        smoothness_error = st.text_input('smoothness error')
        
    #Fourth row
    
        
    with col1:
       compactness_error = st.text_input('compactness_error:')

    with col2:
        concavity_error = st.text_input('concavity_error ')
        
    with col3:
        concave_points_error = st.text_input('concave_points_error')
                                      
    with col4:
        symmetry_error = st.text_input('symmetry_error ')

    with col5:
        fractal_dimension_error = st.text_input('fractal_dimension_error')
        
    #fifth
        
    with col1:
        worst_radius = st.text_input('worst radius (5-30):')

    with col2:
        worst_texture = st.text_input('worst texture ')
        
    with col3:
        worst_perimeter = st.text_input('worst perimeter')
                                      
    with col4:
        worst_area = st.text_input('worst area ')

    with col5:
        worst_smoothness = st.text_input('worst smoothness')
        
    #sixth
   
        
    with col1:
        worst_compactness = st.text_input('worst_compactness')

    with col2:
        worst_concavity = st.text_input('worst concavity')
        
    with col3:
        worst_concave_points = st.text_input('worst_symmetry')
                                      
    with col4:
        worst_symmetry = st.text_input('worst_symmetry ')

    with col5:
        worst_fractal_dimension = st.text_input('worst fractal dimension')

   
    
    #code for prediction
    user_input = ''
    
    #creating a button for prediction
    if st.button("Test Result"):
        user_input = [mean_radius,mean_texture,mean_perimeter,
                                              mean_area,mean_smoothness,mean_compactness,
                                              mean_concavity,mean_concave_points,mean_symmetry,
                                              mean_fractal_dimension,
                                              radius_error,texture_error,perimeter_error,
                                              area_error,smoothness_error,
                                              compactness_error,concavity_error,concave_points_error,
                                              symmetry_error,fractal_dimension_error,
                                              worst_radius,worst_texture,worst_perimeter,
                                              worst_area,worst_smoothness,worst_compactness,
                                              worst_concavity,worst_concave_points,worst_symmetry,
                                              worst_fractal_dimension
                                              ]
    
    user_input = [float(x) for x in user_input]
    diagnosis = breast_cancer_prediction(user_input)
    
        
    st.success(diagnosis)
        
    
if __name__ == '__main__':
    main()
    

    