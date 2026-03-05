import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

#  Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

#  Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

#  Reshape for CNN
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

#  Build CNN model
model = models.Sequential([
    
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=10)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# Predict sample
predictions = model.predict(x_test)

plt.imshow(x_test[0].reshape(28,28), cmap="gray")
plt.title("Predicted: " + str(predictions[0].argmax()))
plt.show()