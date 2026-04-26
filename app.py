import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('xg_model.pkl', 'rb'))

st.title("HR Employee Attrition Prediction")

age = st.number_input('Age', 18, 60, 30)
monthly_income = st.number_input('Monthly Income', 1000, 200000, 5000)
distance = st.number_input('Distance From Home', 1, 50, 5)
years = st.number_input('Years at Company', 0, 40, 5)

gender = st.selectbox('Gender', ('Male', 'Female'))
marital = st.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))
overtime = st.selectbox('OverTime', ('Yes', 'No'))
job_role = st.selectbox('Job Role', (
    'Sales Executive', 'Research Scientist', 'Laboratory Technician',
    'Manufacturing Director', 'Healthcare Representative',
    'Manager', 'Sales Representative', 'Research Director', 'Human Resources'
))

input_dict = {
    'Age': age,
    'MonthlyIncome': monthly_income,
    'DistanceFromHome': distance,
    'YearsAtCompany': years,
    'Gender': gender,
    'MaritalStatus': marital,
    'OverTime': overtime,
    'JobRole': job_role
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error("Employee likely to leave")
    else:
        st.success("Employee will stay")