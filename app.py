import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model  # Assuming you'll use a model later
import streamlit as st
import yfinance as yf

start = '2014-03-05'
end = '2024-03-05'

st.title('Stock Market Navigator')

user_input = st.text_input('Enter Stock Symbol : ')

# Download historical data, handle potential errors
try:
  df = yf.download(user_input, start=start, end=end)
except Exception as e:  # Catch any errors that might occur
  st.error(f"Error downloading data: {e}")
  df = None  # Set df to None to avoid errors in plotting

# Display data only if downloaded successfully
if df is not None:
  st.subheader('Data from 2014 - 2024')
  st.write(df.describe())

  st.subheader('Closing Price vs Time Chart')
  fig = plt.figure(figsize=(12, 6))
  plt.plot(df.Close)
  st.pyplot(fig)

  st.subheader('Closing Price vs Time Chart with 100MA')
  # Calculate 100-day moving average using rolling and reset_index
  ma100_df = df.Close.rolling(100).mean().reset_index()
  plt.plot(ma100_df['Close'])
  plt.plot(df.Close)
  st.pyplot(fig)

  st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
  # Calculate 200-day moving average using rolling and reset_index
  ma200_df = df.Close.rolling(200).mean().reset_index()
  plt.plot(ma100_df['Close'])
  plt.plot(ma200_df['Close'])
  plt.plot(df.Close)
  st.pyplot(fig)


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)

model = load_model("keras_model.h5")

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index = True)

x_test = []
y_test = []
 
for i in range (100, data_training_array.shape[0]):
    x_test.append(data_training_array[i-100: i ])
    y_test.append(data_training_array[i, 0])

    
x_test = np.array(x_test)
y_test =  np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted  = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader("Predictions Vs Original ")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


##Algorithm 
##---> XG Boost ; Light GBM
#  Regression
# Absolute Error & RNN should use while testing