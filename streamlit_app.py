
#Load libraries needed
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from streamlit_lottie import st_lottie
from datetime import datetime

# Load the encoders from files
# LE = joblib.load('Encoders/label_encoder.pkl')
# BE = joblib.load('Encoders/binary_encoder.pkl')
# OE = joblib.load('Encoders/ordinal_encoder.pkl')
# configuration = joblib.load("saved_model/configuration.pkl")
# dataset = joblib.load("saved_model/dataset.pkl")   
test = pd.read_csv("saved_model/test.csv")   

# Load the model
# model = joblib.load('Model/model.pkl')
model = NeuralForecast.load("saved_model/nbeats_0.ckpt")

#define app section
header=st.container()
prediction=st.container()

# Define the Lottie animation URL
lottie_animation_url = "https://lottie.host/89f1f8df-aa47-4771-9441-91da251470e2/qGrHDGTqFH.json"

#define header
with header:
    header.title("London Energy Consumption Prediction")

    # Display the Lottie animation using st_lottie
    st_lottie(lottie_animation_url,height=200)

    header.write("On this page, you can predict the expected energy consumption over the next 7 days.")


# Create lists
inputs = ["num_days", "num_time"]
# categorical = ["holiday", "locale", "transferred"]


# Set up prediction container
with st.expander("Make a prediction", expanded=True):
    
    # Define Streamlit inputs
    # date = st.date_input(label="Enter a date")
    # holiday = st.selectbox(label="Select a category of holiday", options=['Holiday', 'Not Holiday', 'WorkDay', 'Additional', 'Event', 'Transfer', 'Bridge'])
    # locale = st.radio(label="Select a holiday type", options=['National', 'Not Holiday', 'Local', 'Regional'], horizontal=True)
    # transferred = st.radio(label="Select whether the holiday was transferred or not", options=["True", "False"], horizontal=True)
    # onpromotion = st.number_input(label="Please enter the total number of expected items to be on promotions")
    num_days = st.number_input(label="Please enter the number of days to look ahead.")
    num_time = st.number_input(label="Please enter the hour of the day to predict.")

    # Create a button
    predicted = st.button("Predict")

    # Flag variable to control visibility of the prediction message
    show_prediction_message = False

    # Upon predicting
    if predicted:
        show_prediction_message = True  # Set the flag to True when the "Predict" button is pressed

        # Format for inputs
        # input_dict = {
        #     "date": [date],
        #     "holiday": [holiday],
        #     "locale": [locale],
        #     "transferred": [transferred],
        #     "onpromotion": [onpromotion]
        # }

        # Convert inputs into a DataFrame
        # input_df = pd.DataFrame.from_dict(input_dict)

        # Encode categorical inputs
        # input_df["transferred"] = LE.transform(input_df["transferred"])
        # input_df = BE.transform(input_df)
        # input_df["locale"] = OE.transform(input_df[["locale"]])

        # # Convert date to datetime
        # input_df["date"] = pd.to_datetime(input_df["date"])

        # Extract date features and add to input_df
        # input_df['year'] = input_df['date'].dt.year
        # input_df['month'] = input_df['date'].dt.month
        # input_df['day'] = input_df['date'].dt.day
        # input_df['day_of_week'] = input_df['date'].dt.dayofweek
        # input_df['day_of_year'] = input_df['date'].dt.dayofyear
        # input_df['week_of_year'] = input_df['date'].dt.isocalendar().week
        # input_df['quarter'] = input_df['date'].dt.quarter
        # input_df['is_weekend'] = (input_df['date'].dt.dayofweek // 5 == 1).astype(int)
        # input_df['day_of_month'] = input_df['date'].dt.day

        # Drop date after extracting
        # input_df.drop(columns=['date'], inplace=True)

        # Make predictions
        model_output = model.predict(futr_df=test).reset_index()
        predict_dict = prediction[['ds', 'NBEATS']].to_dict()

        if UserInputTime == '':
            row = int(num_days)*24 - 1
        else:
            row = int(num_days)*24 - 1 + num_time - 1

        predicted_value = predict_dict['NBEATS'][row]

        # Round the model output to 3 dec pl
        rounded_output = np.round(predicted_value, 3)

        # Add rounded predictions to the input_dict
        # input_dict["Total Sales($)"] = rounded_output

        # Format the rounded output with bold and a dollar sign
        # formatted_output = f"<b>${rounded_output[0]}</b>"

# Display prediction message inside the expander with HTML formatting
        st.write(f"The predicted energy consumption is {rounded_output}", unsafe_allow_html=True)


# Display the DataFrame outside the expander
if show_prediction_message:
    st.write("A dataframe containing inputs and your predicted sales is shown below:")
    st.dataframe(input_dict)
