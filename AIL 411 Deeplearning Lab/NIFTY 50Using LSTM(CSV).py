
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------- 1. Load CSV ----------
data = pd.read_csv("nifty50.csv", skiprows=3, header=None,
                   usecols=[0, 2], names=["Date", "Close"])

dates = pd.to_datetime(data["Date"])
prices = data["Close"].values.reshape(-1, 1)
# ---------- 2. Normalize ----------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(prices)

# ---------- 3. Create sequences ----------
sequence_length = 60

X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X).reshape(-1, sequence_length, 1)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ---------- 4. Train / Test Split ----------
train_size = int(len(X) * 0.80)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------- 5. Build and Train LSTM ----------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=20,
          batch_size=32, verbose=1)

# ---------- 6. Prediction ----------
prediction = scaler.inverse_transform(
    model.predict(X_test, verbose=0)
).flatten()

actual = prices[sequence_length + train_size:].flatten()
test_dates = dates.iloc[sequence_length + train_size:]

# ---------- 7. Evaluation ----------
mae = mean_absolute_error(actual, prediction)

print("\nMean Absolute Error:", mae)

# ---------- 8. Plot ----------
plt.figure(figsize=(16, 8))

plt.plot(dates, prices.flatten(), label="All Data", alpha=0.45)
plt.plot(test_dates, actual, "--", label="True Values")
plt.plot(test_dates, prediction, "--", label="Predicted Values")
plt.title("NIFTY 50 LSTM Prediction")
plt.xlabel("Date")
plt.ylabel("NIFTY 50 Close Price")

plt.show()
