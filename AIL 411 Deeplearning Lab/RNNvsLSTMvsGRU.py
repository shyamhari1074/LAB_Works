import time
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, LSTM, GRU, Dense

vocab_size, max_length = 10000, 200

# ---------- 1. Load and pad data ----------
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

x_train = pad_sequences(x_train, maxlen=max_length, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=max_length, padding="post", truncating="post")
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# ---------- 2. Train each model ----------
architectures = {"Simple RNN": SimpleRNN, "LSTM": LSTM, "GRU": GRU}
results = []   # (name, accuracy %, loss, time)

for name, recurrent_layer in architectures.items():
    print("\nTraining model:", name)

    model = Sequential([
        Input(shape=(max_length,)),
        Embedding(vocab_size, 64),
        recurrent_layer(64),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    start = time.time()
    model.fit(x_train, y_train, epochs=6, batch_size=64, validation_split=0.2)
    train_time = time.time() - start

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    results.append((name, round(accuracy * 100, 2), round(loss, 4), round(train_time, 2)))

# ---------- 3. Comparison table ----------
print("\n" + "=" * 75)
print("RNN, LSTM AND GRU PERFORMANCE COMPARISON")
print("=" * 75)
print(f"{'Model':<15}{'Test Accuracy (%)':>20}{'Test Loss':>15}{'Training Time (s)':>23}")
print("-" * 75)

for name, accuracy, loss, train_time in results:
    print(f"{name:<15}{accuracy:>20.2f}{loss:>15.4f}{train_time:>23.2f}")

print("=" * 75)

# ---------- 4. Final analysis ----------
best_accuracy_model = max(results, key=lambda r: r[1])[0]
lowest_loss_model = min(results, key=lambda r: r[2])[0]

print("\n" + "=" * 35)
print("FINAL ANALYSIS")
print("=" * 35)
print(f"Best Test Accuracy : {best_accuracy_model}")
print(f"Lowest Test Loss   : {lowest_loss_model}")
print("=" * 35)
