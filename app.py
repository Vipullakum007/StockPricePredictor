import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime

# App Config
st.set_page_config(page_title='Stock Trend Prediction', layout='wide')
st.title('Stock Trend Prediction App')

# Sidebar Inputs
st.sidebar.header("Select Options")
user_input = st.sidebar.text_input('Enter Stock Ticker:', 'AAPL').strip().upper()
attribute = st.sidebar.selectbox('Select Attribute:', ['Close', 'Open', 'High', 'Low'])

# Date Selection
st.sidebar.subheader("Select Date Range ( select big range for better analysis ) ")
start_date = st.sidebar.date_input("Select Start Date", datetime(2015, 1, 1))
end_date = st.sidebar.date_input("Select End Date", datetime(2025, 1, 1))

# Ensure end date is not before start date
if start_date >= end_date:
    st.error("End date must be after the start date.")
    st.stop()

if not user_input:
    st.error("Invalid stock ticker. Please enter a valid symbol.")
    st.stop()

# Fetch Stock Name
try:
    stock = yf.Ticker(user_input)
    stock_name = stock.info.get('longName', 'Unknown Stock')
except Exception:
    stock_name = "Unknown Stock"

# Display Stock Name
st.subheader(f"Selected Stock: {stock_name} ({user_input})")

@st.cache_data
def get_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

with st.spinner("Fetching data..."):
    df = get_stock_data(user_input, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

# Validate Data
if df.empty or attribute not in df.columns:
    st.error("No data found! Please enter a valid stock ticker.")
    st.stop()

# Clean Missing Data
df.fillna(method='ffill', inplace=True)

# Display Data Summary
st.subheader(f'{attribute} Price Data')
st.write(df.describe())

# Visualization
st.subheader(f'{attribute} Price Trend')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df[attribute], label=attribute, color='b')
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Moving Averages
st.subheader(f'{attribute} Price with Moving Averages')
ma100, ma200 = df[attribute].rolling(100).mean(), df[attribute].rolling(200).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ma100, 'g', label='100-day MA', alpha=0.8)
ax.plot(ma200, 'r', label='200-day MA', alpha=0.8)
ax.plot(df[attribute], 'b', label=attribute, alpha=0.6)
ax.legend()
st.pyplot(fig)

# Data Splitting
train_size = int(len(df) * 0.70)
data_training, data_testing = df[[attribute]][:train_size], df[[attribute]][train_size:]

# Scaling Data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Preparing Training Data
x_train, y_train = [], []
for i in range(100, len(data_training_array)):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Load Model Safely
model_path = 'Keras_model.h5'
if not os.path.exists(model_path):
    st.error("Model file not found! Please check the path.")
    st.stop()

model = load_model(model_path)

# Preparing Testing Data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []
for i in range(100, len(input_data)):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Making Predictions
with st.spinner("Making Predictions..."):
    y_predicted = model.predict(x_test)

# Rescaling Predictions
scale_factor = 1 / scaler.scale_[0]
y_predicted, y_test = y_predicted * scale_factor, y_test * scale_factor

# Predictions vs Original
st.subheader(f'Predictions vs Actual {attribute} Price')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test, 'b', label='Actual Price')
ax.plot(y_predicted, 'r', linestyle='dashed', label='Predicted Price')
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

st.success("Prediction Completed!")
