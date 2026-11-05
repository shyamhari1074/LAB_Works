import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

sequence_length = 60
# ---------- 1. Download NIFTY 50 data ----------
data = yf.download("^NSEI", start="2015-01-01", end="2026-01-01", auto_adjust=False)
close = data["Close"].squeeze().dropna()   # one column -> Series

dates = close.index
prices = close.values.reshape(-1, 1)
print("Total records:", len(prices))

# ---------- 2. Normalize ----------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# ---------- 3. Create sequences (60 days -> next day) ----------
X = np.array([scaled_data[i - sequence_length:i, 0]
              for i in range(sequence_length, len(scaled_data))])
X = X.reshape(X.shape[0], X.shape[1], 1)   # LSTM needs 3D input
y = scaled_data[sequence_length:, 0]

print("X shape:", X.shape)
print("y shape:", y.shape)

# ---------- 4. Train / test split (80 / 20) ----------
train_size = int(len(X) * 0.80)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------- 5. Build, compile and train LSTM ----------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# ---------- 6. Predict and convert back to NIFTY prices ----------
prediction = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
actual = prices[sequence_length + train_size:].flatten()

# ---------- 7. MAE ----------
mae = mean_absolute_error(actual, prediction)
print("\nMean Absolute Error:", mae)

# ---------- 8. Plot ----------
test_dates = dates[sequence_length + train_size:]

plt.figure(figsize=(16, 8))
plt.plot(dates, prices.flatten(), label="All Data", alpha=0.45)
plt.plot(test_dates, actual, linestyle="--", label="True Values")
plt.plot(test_dates, prediction, linestyle="--", label="Predicted Values")
plt.axvline(test_dates[0], linestyle="--", linewidth=2, label="Train / Test Split")

plt.title("NIFTY 50 LSTM Prediction")
plt.xlabel("Date")
plt.ylabel("NIFTY 50 Close Price")
plt.text(0.02, 0.94, f"MAE: {mae:.2f}", transform=plt.gca().transAxes, fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
