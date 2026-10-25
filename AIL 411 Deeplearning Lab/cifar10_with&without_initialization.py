import tensorflow as tf
import time
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

epochs, batch_size = 5, 64

def build_model(initializer="glorot_uniform", dropout=False, regularizer=None):
    layers = [Flatten(input_shape=(32, 32, 3))]
    for units in [512, 256, 64]:
        layers.append(Dense(units, activation="relu",
                            kernel_initializer=initializer,
                            kernel_regularizer=regularizer))
        if dropout:
            layers.append(Dropout(0.5))
    layers.append(Dense(10, activation="softmax", kernel_initializer=initializer))
    return Sequential(layers)

models = {
    "Baseline":          dict(initializer="glorot_uniform"),
    "Xavier":            dict(initializer="glorot_uniform"),
    "Kaiming":           dict(initializer="he_normal"),
    "Dropout":           dict(dropout=True),
    "L2 Regularization": dict(regularizer=l2(0.001)),
}

# "adam" and "sgd" use Keras defaults: Adam lr=0.001, SGD lr=0.01
optimizers = ["Adam", "SGD"]

results = []

for model_name, config in models.items():
    for opt_name in optimizers:
        model = build_model(**config)
        model.compile(optimizer=opt_name.lower(),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        start = time.time()
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(x_test, y_test), verbose=1)
        train_time = time.time() - start

        loss, acc = model.evaluate(x_test, y_test, verbose=0)

        reg = ("Dropout" if config.get("dropout")
               else "L2" if config.get("regularizer") else "None")

        results.append([model_name, opt_name, reg, loss, acc * 100, train_time])

results_df = pd.DataFrame(results, columns=[
    "Model", "Optimizer", "Regularization",
    "Test Loss", "Test Accuracy (%)", "Training Time (seconds)"
])

print("\n" + "=" * 100)
print("FINAL COMPARISON")
print("=" * 100)
print(results_df.to_string(index=False))
