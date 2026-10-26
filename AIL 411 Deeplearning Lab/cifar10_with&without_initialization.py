
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Load and prepare data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

def build_model(init, dropout, reg):
    model = Sequential([Flatten(input_shape=(32, 32, 3))])
    for units in [512, 256, 64]:
        model.add(Dense(units, activation="relu", kernel_initializer=init, kernel_regularizer=reg))
        if dropout:
            model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax", kernel_initializer=init))
    return model

# (name, initializer, dropout, regularizer, regularization label)
experiments = [
    ("Baseline",          "glorot_uniform", False, None,       "None"),
    ("Xavier",            "glorot_uniform", False, None,       "None"),
    ("Kaiming",           "he_normal",      False, None,       "None"),
    ("Dropout",           "glorot_uniform", True,  None,       "Dropout"),
    ("L2 Regularization", "glorot_uniform", False, l2(0.001),  "L2"),
]

results = []

for name, init, dropout, reg, reg_label in experiments:
    for opt in ["Adam", "SGD"]:
        model = build_model(init, dropout, reg)
        model.compile(optimizer=opt.lower(), loss="categorical_crossentropy", metrics=["accuracy"])

        start = time.time()
        model.fit(x_train, y_train, epochs=5, batch_size=64,
                  validation_data=(x_test, y_test))
        train_time = time.time() - start

        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        results.append([name, opt, reg_label, loss, acc * 100, train_time])

results_df = pd.DataFrame(results, columns=[
    "Model", "Optimizer", "Regularization",
    "Test Loss", "Test Accuracy (%)"
])

print("\n" + "=" * 100)
print("FINAL COMPARISON")
print("=" * 100)
print(results_df.to_string(index=False))
