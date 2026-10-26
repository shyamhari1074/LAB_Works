import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load and prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

def build_model(extra_conv):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2))
    ])
    if extra_conv:
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    return model

# Model 1 has one conv layer, Model 2 has two
for number, extra_conv in [(1, False), (2, True)]:
    model = build_model(extra_conv)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=64)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    pred = model.predict(x_test[:2])

    print("\n==============================")
    print(f"MODEL {number} RESULTS")
    print("==============================")
    print("Accuracy:", accuracy * 100, "%")
    print("Loss:", loss)

    plt.figure(figsize=(12,6))
for i in range(2):
    one_hot=str(ytest[i].astype(int).tolist())
    plt.subplot(1,2,i+1)
    plt.imshow(xtest[i],cmap='gray')
    plt.title(f"Predictecd:{pred[i].argmax()}\n"
             f"Actual :{ytest[i].argmax()}\n"
             f"One hot:{one_hot}")
        plt.axis("off")

    plt.suptitle(f"Model {number} Predictions")
    plt.tight_layout()
    plt.show()
