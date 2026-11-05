import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import to_categorical

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def prepare(x):
    x = tf.image.resize(x[..., None], (32, 32))  # add channel, resize 28x28 -> 32x32
    return preprocess_input(tf.image.grayscale_to_rgb(x))

x_train, x_test = prepare(x_train), prepare(x_test)
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

def build_model(fine_tune):
    base = VGG19(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    # Fixed feature extraction freezes everything; fine-tuning keeps the last 4 layers trainable
    for layer in (base.layers[:-4] if fine_tune else base.layers):
        layer.trainable = False
    x = layers.Flatten()(base.output)
    x = layers.Dense(512, activation="relu")(x)
    output = layers.Dense(10, activation="softmax")(x)
    return Model(base.input, output)

runs = {"Fixed Feature Extraction": False, "Fine-Tuning": True}
results = {}

for name, fine_tune in runs.items():
    model = build_model(fine_tune)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001) if fine_tune else "adam"
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    start = time.time()
    history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    train_time = time.time() - start

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    results[name] = (acc, loss, train_time, history.history)

    print(name.upper())
    print("Test Accuracy:", acc * 100, "%")
    print("Test Loss:", loss)
    print("Training Time:", train_time, "seconds")

# Plot accuracy and loss
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for row, (name, result) in enumerate(results.items()):
    hist = result[3]
    for col, metric in enumerate(["Accuracy", "Loss"]):
        ax = axes[row, col]
        ax.plot(hist[metric.lower()], marker="o", label="Training " + metric)
        ax.plot(hist["val_" + metric.lower()], marker="o", label="Validation " + metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{name} - {metric}")
        ax.legend()
        ax.grid()

plt.tight_layout()
plt.show()
