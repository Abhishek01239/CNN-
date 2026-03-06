import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train = X_train /255.0
X_test = X_test/255.0

model = keras.Sequential([

    layers.Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3),activation="relu"),

    layers.Flatten(),

    layers.Dense(128,activation="relu"),
    layers.Dense(10,activation="softmax")
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

history = model.fit(
    X_train,
     y_train,
     epochs =10,
     validation_data = (X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:",accuracy)

plt.plot(history.history['accuracy'], label = 'train')
plt.plot(history.history['val_accuracy'], label = 'validation')
plt.legend()
plt.title("Training Accuracy")
plt.show()