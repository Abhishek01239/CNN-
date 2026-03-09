import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load dataset (example using CIFAR10 as placeholder)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Convert to binary labels (tumor / no tumor simulation)
y_train = (y_train % 2)
y_test = (y_test % 2)

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# CNN Model
model = keras.Sequential([

    layers.Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3),activation="relu"),

    layers.Flatten(),

    layers.Dense(128,activation="relu"),
    layers.Dense(1,activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test,y_test)
)

# Evaluate
loss,acc = model.evaluate(x_test,y_test)

print("Test Accuracy:",acc)

# Plot accuracy
plt.plot(history.history["accuracy"],label="train")
plt.plot(history.history["val_accuracy"],label="validation")
plt.legend()
plt.title("Training Accuracy")
plt.show()