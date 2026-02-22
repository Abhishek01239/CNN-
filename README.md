# 🧠 CNN Project — Handwritten Digit Recognition

This project implements a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) using the MNIST dataset.

It is a beginner-friendly deep learning project and a starting point for computer vision.

---

## 🚀 Project Overview

The model:

- Takes handwritten digit images as input
- Extracts features using convolution layers
- Predicts the correct digit

This project demonstrates how CNNs work in image classification tasks.

---

## 🎯 Objectives

- Learn Convolutional Neural Networks (CNN)
- Understand image preprocessing
- Train deep learning models
- Perform multi-class classification
- Evaluate model performance

---

## 📂 Dataset

### MNIST Dataset

- 60,000 training images
- 10,000 testing images
- 28×28 grayscale images
- 10 classes (digits 0–9)

Dataset loads automatically using TensorFlow.

---

## 🏗️ Model Architecture

Input (28×28×1)
→ Conv2D (32 filters)
→ MaxPooling
→ Conv2D (64 filters)
→ MaxPooling
→ Flatten
→ Dense (64)
→ Output (10 classes)

---

## ⚙️ Installation

### Clone repository

git clone https://github.com/Abhishek01239/CNN-.git
cd CNN-

### Install dependencies

pip install tensorflow matplotlib

---

## ▶️ Run Project

python main.py

---

## 📊 Results

- Training Accuracy: ~99%
- Testing Accuracy: ~98%

---

## 🧩 Project Structure

CNN/
│
├── main.py
├── README.md
└── requirements.txt

---

## 🔧 Future Improvements

- Add dropout layer
- Train longer epochs
- Use custom handwritten images
- Deploy as web app
- Improve accuracy

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 👨‍💻 Author

Abhishek
