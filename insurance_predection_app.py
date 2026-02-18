import streamlit as st 
from joblib import load
import pandas as pd 
import numpy as np

# Provide the correct file path to the location where the dataset is stored 
df = pd.read_csv(r"C:\Users\hp5cd\Documents\machine learning\capstone project\Data\data.csv")

# load trained model 
ml_model = load(r"C:\Users\hp5cd\Documents\machine learning\capstone project\model file\RF_model.joblib")

lower_bound, upper_bound = load(r"C:\Users\hp5cd\Documents\machine learning\capstone project\model file\premium_bounds.joblib")

model_columns = load(r"C:\Users\hp5cd\Documents\machine learning\capstone project\model file\model_columns.joblib")

# set title for UI
st.title("Insurance Customer Response Prediction")

# input filed of UI
st.header("Enter Customer Details")
gender = ['Male', 'Female']
age = np.arange(18, 50, 1)
driving_license = ['Yes', 'No']
region_code = sorted(df['Region_Code'].unique())
previously_insured = ['Yes', 'No']
vehicle_age = sorted(df['Vehicle_Age'].unique())
vehicle_damage = ['Yes', 'No']
sales_channel = sorted(df['Policy_Sales_Channel'].unique())

Gender = st.selectbox("Gender", gender)
Age = st.selectbox("Age", age)
Driving_license = st.selectbox("Driving License", driving_license)
Region_code = st.selectbox("Select Region Code", region_code)
Previously_Insured = st.selectbox("Previously Insured", previously_insured)
Vehicle_Age = st.selectbox("Vehicle Age", vehicle_age)
Vehicle_Damage = st.selectbox("Vehicle Damage", vehicle_damage)
Annual_Premium = st.number_input("Annual Premium")
Sales_Channel = st.selectbox("Sales Channel", sales_channel)
Vintage = st.number_input("Vintage", min_value=0)

Gender = 0 if Gender == 'Male' else 1
Driving_license = 1 if Driving_license == 'Yes' else 0
Previously_Insured = 1 if Previously_Insured == 'Yes' else 0
Vehicle_Damage = 1 if Vehicle_Damage == 'Yes' else 0

Annual_Premium = np.clip(Annual_Premium, lower_bound, upper_bound)

input_df = pd.DataFrame({
    'Gender': [Gender],
    'Age': [Age],
    'Driving_License': [Driving_license],
    'Region_Code': [Region_code],
    'Previously_Insured': [Previously_Insured],
    'Vehicle_Age': [Vehicle_Age],
    'Vehicle_Damage': [Vehicle_Damage],
    'Annual_Premium': [Annual_Premium],
    'Policy_Sales_Channel': [Sales_Channel],
    'Vintage': [Vintage]
})

input_df = pd.get_dummies(input_df)
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]

if st.button('predict'):
    prediction = ml_model.predict(input_df)[0]
    probability = ml_model.predict_proba(input_df)[0][1]

    st.subheader('Prediction Result')
    if prediction == 1:
        st.success('Yes')
    else:
        st.error('No')
    st.write('Response Probability :', round(probability, 3))