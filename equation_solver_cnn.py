import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train /255.0
X_test = X_test/255.0

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

model = keras.Sequential([
    layers.Conv2D(32,(3,3), activation ="relu", input_shape =(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3), activation = "relu"),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(128, activation ="relu"),
    layers.Dense(10, activation = "softmax")
])

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

model.fit(X_train, y_train, epochs =5, validation_data = (X_test, y_test))

equation = "3+5"

result = eval(equation)

print("Equation: ", equation)
print("Result: ", result)