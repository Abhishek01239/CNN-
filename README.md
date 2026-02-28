# 🧠 CNN Projects Collection

This repository contains multiple Convolutional Neural Network (CNN) projects built using TensorFlow.

These projects demonstrate my learning journey in Deep Learning and Computer Vision.

---

# 📌 Projects Included

---

## 1️⃣ MNIST Handwritten Digit Recognition

### 📖 Description
Classifies handwritten digits (0–9) using CNN.

### 📊 Dataset
Built-in dataset from TensorFlow  
- 60,000 training images  
- 10,000 testing images  
- 28x28 grayscale images  

### 🎯 Accuracy
~98–99%

---

## 2️⃣ Cats vs Dogs Image Classifier

### 📖 Description
Binary image classification model that predicts whether an image is a cat or dog.

### 📊 Dataset
- TensorFlow Datasets (cats_vs_dogs)

### 🎯 Accuracy
~85–95%

---

## 3️⃣ Fashion MNIST Image Classifier

### 📖 Description
Classifies clothing images into 10 categories such as:
T-shirt, Trouser, Dress, Sneaker, Bag, Coat, etc.

### 📊 Dataset
Built-in Fashion MNIST dataset from TensorFlow.

### 🎯 Accuracy
~88–92%

---

## 4️⃣ CIFAR-10 Image Classification (Advanced CNN)

### 📖 Description
Multi-class image classification model trained on real-world colored images.

### 📊 Dataset
CIFAR-10 dataset (built-in TensorFlow dataset)

- 60,000 color images
- 32x32 RGB images
- 10 classes:
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

### 🧠 Model Architecture
Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → MaxPooling →  
Flatten → Dense → Dropout → Output (Softmax)

### 🎯 Accuracy
~70–80%

---

# 🏗 Project Structure
CNN/
│
├── mnist_cnn.py
├── cat_dog_cnn.py
├── fashion_cnn.py
├── cifar10_cnn.py
├── README.md
└── requirements.txt


---

# ⚙️ Installation

Clone repository:
git clone https://github.com/Abhishek01239/CNN-.git

cd CNN-

Install dependencies:
pip install tensorflow tensorflow-datasets matplotlib numpy

---

# ▶️ How To Run

Run any project:
python mnist_cnn.py
python cat_dog_cnn.py
python fashion_cnn.py
python cifar10_cnn.py


---

# 🛠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

# 📚 What I Learned

- Convolutional Neural Networks (CNN)
- Binary Classification
- Multi-class Classification
- RGB Image Processing
- Dropout Regularization
- Model Evaluation
- Training vs Validation Accuracy Visualization

---

# 🚀 Future Improvements

- Data Augmentation
- Transfer Learning (MobileNet / ResNet)
- Confusion Matrix Visualization
- Model Deployment (Flask / Streamlit)
- Real-time Image Prediction
- Save and Load Trained Models

---

# 👨‍💻 Author

Abhishek

---

⭐ If you found this repository useful, consider giving it a star!