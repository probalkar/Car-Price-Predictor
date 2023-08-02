import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

def load_car_models(company):
    car_models = car[car['company'] == company]['name'].unique()
    return car_models

def main():
    st.set_page_config(page_title="Car Price Predictor", layout="wide")
    st.title("Car Price Predictor")

    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    st.write("This app predicts the price of a car you want to sell. Try filling the details below:")

    company = st.selectbox("Select the company:", companies, index=0)
    if company != "Select Company":
        car_models = load_car_models(company)
        car_model = st.selectbox("Select the model:", car_models)
    year = st.selectbox("Select Year of Purchase:", years)
    fuel_type = st.selectbox("Select the Fuel Type:", fuel_types)
    kilo_driven = st.text_input("Enter the Number of Kilometres that the car has travelled:")

    if st.button("Predict Price"):
        if company != "Select Company" and car_model:
            prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                              data=np.array([car_model, company, year, kilo_driven, fuel_type]).reshape(1, 5)))
            st.title(f"Prediction: ₹{np.round(prediction[0], 2)}")

if __name__ == '__main__':
    main()
