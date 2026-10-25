import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

def create_model(hidden_units, activation):
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(hidden_units[0], activation=activation),
        Dense(hidden_units[1], activation=activation),
        Dense(hidden_units[2], activation=activation),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

hidden_units_list = [
    (1024, 512, 256),
    (512, 256, 128),
    (256, 128, 64),
    (128, 64, 32)
]

activation_list = [
    "relu",
    "tanh",
    "sigmoid"
]

results = []

best_acc = 0
best_model = None
best_config = None

for units in hidden_units_list:
    for act in activation_list:
        model = create_model(units, act)
        model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)

        results.append([*units, act, loss, acc * 100])

        if acc > best_acc:
            best_acc, best_model, best_config = acc, model, (units, act)

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Hidden Layer 1':>15}"
    f"{'Hidden Layer 2':>15}"
    f"{'Hidden Layer 3':>15}"
    f"{'Activation':>12}"
    f"{'Test Loss':>12}"
    f"{'Test Accuracy (%)':>20}"
)

for result in results:
    print(
        f"{result[0]:>15}"
        f"{result[1]:>15}"
        f"{result[2]:>15}"
        f"{result[3]:>12}"
        f"{result[4]:>12.4f}"
        f"{result[5]:>20.2f}"
    )

print()
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Hidden Units :", best_config[0])
print("Activation   :", best_config[1])
print("Accuracy     :", f"{best_acc * 100:.2f} %")

plt.figure(figsize=(15, 6))

for i in range(10):
    prediction = best_model.predict(x_test[i].reshape(1, 32, 32, 3), verbose=0)
    predicted_class = np.argmax(prediction)
    actual_class = np.argmax(y_test[i])

    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(
        f"Pred: {class_names[predicted_class]}\n"
        f"Actual: {class_names[actual_class]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()
