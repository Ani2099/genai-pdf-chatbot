# yfinance: Used to fetch historical stock price data.

# numpy & pandas: For data manipulation.

# matplotlib: To plot graphs.

# MinMaxScaler: Scales stock prices between 0 and 1 for better model training.

# tensorflow.keras: To build and train the LSTM deep learning model.

import yfinance as yf
data = yf.download('GOOGL', start='2020-01-01', end='2025-12-27')
closing_prices = data['Close'].values.reshape(-1, 1)
data.head()


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(closing_prices)

X = []
y = []
seq_len = 60  # Use last 60 days

for i in range(seq_len, len(scaled_prices)):
    X.append(scaled_prices[i - seq_len:i, 0])
    y.append(scaled_prices[i, 0])

import numpy as np
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # 3D shape for LSTM


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)


1. Creating the Model
python
Copy
Edit
model = Sequential()
Instantiates a Sequential model object.

We will add layers to this model one by one.

🔸 2. First LSTM Layer
python
Copy
Edit
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
Adds an LSTM layer with:

units=50: Number of LSTM units/neurons (each has its own internal memory).

return_sequences=True: Important! This tells the layer to return the entire sequence of outputs for each input time step, not just the last output. This is required because the next LSTM layer also expects sequences.

input_shape=(X.shape[1], 1): Shape of each training example:

X.shape[1]: Number of time steps in each input sample.

1: Number of features (only one in this case, e.g., stock price per day).

🔸 3. Second LSTM Layer
python
Copy
Edit
model.add(LSTM(units=50))
Another LSTM layer with:

units=50: Again, 50 neurons.

No return_sequences parameter: This means it will only output the last hidden state (a vector), not the whole sequence.

Since this is the final LSTM layer, we want a summary of the sequence, not the full sequence.

🔸 4. Dense Output Layer
python
Copy
Edit
model.add(Dense(1))
Adds a fully connected (Dense) layer with:

1 neuron because we want to predict a single numeric value (e.g., next day's stock price).



To predict the next 30 days prices:

Take the last 60 days of scaled data as input.

Predict the next day’s price.

Append this prediction to the sequence.

Use the updated sequence (including predicted value) to predict the next day.

This is called recursive forecasting.

Finally, scale the predicted values back to original price scale.

Sequential(): Creates a linear stack of layers.

LSTM(units=50, return_sequences=True, ...): First LSTM layer with 50 units; returns sequences for the next LSTM layer.

LSTM(units=50): Second LSTM layer; doesn't return sequences (last layer).

Dense(1): Output layer — predicts one value (the stock price).

compile(...): Uses Adam optimizer and MSE loss (standard for regression).

fit(...): Trains the model for 20 epochs with batch size 32.

# **Predict Future Prices**

last_60_days = scaled_prices[-60:]  # Last 60 days from data
input_data = last_60_days.reshape(1, 60, 1)
predicted_price = model.predict(input_data)
predicted_price_real = scaler.inverse_transform(predicted_price)


scaled_prices[-60:]: Extracts the last 60 days from the full dataset.

reshape(1, 60, 1): Prepares the input in 3D shape for LSTM prediction.

model.predict(...): Predicts the next scaled value.

inverse_transform(...): Converts the normalized prediction back to the original scale (actual stock price).

future_days = 30
predicted_future = []

current_input = scaled_prices[-60:].tolist()

for _ in range(future_days):
    input_seq = np.array(current_input[-60:]).reshape(1, 60, 1)
    next_pred = model.predict(input_seq)[0][0]
    current_input.append([next_pred])
    predicted_future.append(next_pred)

predicted_future = scaler.inverse_transform(np.array(predicted_future).reshape(-1, 1))


# **Plot the Predictions**

Visualize the predicted stock prices for the next 30 days.

The X-axis represents days into the future.

The Y-axis shows predicted stock price values.

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(predicted_future, label='Predicted Future Price')
plt.title('Future Stock Price Prediction (Next 30 Days)')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()
