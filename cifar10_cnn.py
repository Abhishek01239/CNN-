import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train/255.0
X_test = X_test/255.0

model = models.Sequential([
    layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), acivation = 'relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation = 'softmax')
])

model.compile(
    optimizer =  'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
    )

history = model.fit(
    X_train, y_train,
    epochs = 10,validation_data = (X_test, y_test)
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("\n Test Accuracy:", test_acc)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Validation'])
plt.title("Accuracy")
plt.show()