import numpy as np
import streamlit as st

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('Salary.csv')

df = pd.DataFrame(data)

X = data[['YearsExperience']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# r2 = r2_score(y_test, y_pred)               # R² Score

# # Display the results

# print(f"R² Score: {r2:.2f}")

model = LinearRegression()

model.fit(X_train, y_train)

st.title("Salary Prediction App")
st.write("Enter your years of experience to predict your salary:")
years_experience = st.number_input("Years of Experience", min_value=0.0, step=0.1)

# Predict salary
if st.button("Predict"):
    predicted_salary = model.predict([[years_experience]])
    st.write(f"Predicted Salary: ₹ {predicted_salary[0]:,.2f}")




