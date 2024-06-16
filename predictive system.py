# -*- coding: utf-8 -*-


"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C:/Machine-Learning/cancer/cancer_lreg.sav', 'rb'))


input_data = (13.17,18.66,85.98,534.6,0.1158,0.1231,0.1226,0.0734,0.2128,0.06777,0.2871
              ,0.8937,1.897,24.25,0.006532,0.02336,0.02905,0.01215,0.01743,0.003643,15.67
              ,27.95,102.8,759.4,0.1786,0.4166,0.5006,0.2088,0.39,0.1179857155)

#change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardizing the input data
#input_data_std = loaded_scaler.transform(input_data_reshaped)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0]==0):
  print('The tumor is Malignant')
else:
  print('The tumor is Benign')