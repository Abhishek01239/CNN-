import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert to grayscale
x_train_gray = tf.image.rgb_to_grayscale(x_train)
x_test_gray = tf.image.rgb_to_grayscale(x_test)

# CNN Model
model = keras.Sequential([

    layers.Conv2D(64,(3,3),activation="relu",padding="same",input_shape=(32,32,1)),
    layers.Conv2D(64,(3,3),activation="relu",padding="same"),
    
    layers.Conv2D(32,(3,3),activation="relu",padding="same"),

    layers.Conv2D(3,(3,3),activation="sigmoid",padding="same")
])

model.compile(
    optimizer="adam",
    loss="mse"
)

# Train
model.fit(
    x_train_gray,
    x_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test_gray,x_test)
)

# Predict
output = model.predict(x_test_gray[:5])

# Show result
for i in range(5):

    plt.figure(figsize=(8,3))

    plt.subplot(1,3,1)
    plt.title("Gray")
    plt.imshow(x_test_gray[i].squeeze(),cmap="gray")

    plt.subplot(1,3,2)
    plt.title("Original")
    plt.imshow(x_test[i])

    plt.subplot(1,3,3)
    plt.title("Predicted Color")
    plt.imshow(output[i])

    plt.show()