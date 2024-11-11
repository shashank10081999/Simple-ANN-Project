import streamlit as st
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
import pandas as pd 
import pickle


model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl' , 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl' , 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('standard_scaler.pkl' , 'rb') as file:
    standard_scaler = pickle.load(file)


st.title('Customer Prediction')


geograph = st.selectbox('Geograph' , onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender' , label_encoder_gender.classes_)
age = st.slider("Age" , 18 , 25)
balance = st.number_input('Balance')
creditScore = st.slider('CreditScore' , 350 , 850)
tenure = st.slider('Tenure' , 1 ,10)
numOfProducts = st.selectbox('NumOfProducts', range(1 , 5,1))
hasCrCard = int(st.checkbox('HasCrCard'))
is_active_member = int(st.checkbox('IsActive'))
estimated_salary = st.number_input('Estimated Salary')


input_data = {
    'CreditScore': creditScore,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': numOfProducts,
    'HasCrCard': hasCrCard,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
}

geo_data = pd.DataFrame(onehot_encoder_geo.transform([[geograph]]).toarray() , columns = onehot_encoder_geo.get_feature_names_out(["Geography"]))

data = pd.concat([pd.DataFrame([input_data]) , geo_data] , axis=1 )

input_values = standard_scaler.transform(data)

st.write(f"The probability with the customer will churn is - {model.predict(input_values)[0][0]}")






