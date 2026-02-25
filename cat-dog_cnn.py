import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Load dataset
dataset, info = tfds.load(
    "tf_flowers",
    as_supervised=True,
    with_info=True
)

train_data = dataset["train"]

# Resize + normalize
def preprocess(image, label):
    image = tf.image.resize(image, (150,150))
    image = image / 255.0
    return image, label

train_data = train_data.map(preprocess).batch(32)

# Build CNN
model = models.Sequential([
    layers.Conv2D(32,3,activation="relu",input_shape=(150,150,3)),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64,3,activation="relu"),
    layers.MaxPooling2D(),
    
    layers.Conv2D(128,3,activation="relu"),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128,activation="relu"),
    layers.Dropout(0.5),
    
    layers.Dense(5,activation="softmax")   # 5 flower classes
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, epochs=5)