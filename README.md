

# 🧠 Brain Tumor Detection using CNN

This project focuses on the **detection of brain tumors** from grayscale MRI images using **Convolutional Neural Networks (CNN)**. It aims to assist medical professionals by providing an accurate and automated diagnosis tool.

## 📌 Problem Statement

Interpreting brain MRI scans is a complex task prone to human error due to various factors like tumor size and image quality. This project addresses this challenge by implementing a deep learning solution capable of reliable and consistent classification.

## 🎯 Objective

- Detect the presence of brain tumors in grayscale MRI images.
- Achieve high accuracy across diverse image qualities and patient anatomies.

## 🛠️ Technologies Used

- **Python**: Programming language.
- **Google Colab**: Cloud-based training and testing.
- **TensorFlow / Keras**: Deep learning framework.
- **OpenCV, PIL**: Image processing and manipulation.
- **NumPy**: Numerical operations.
- **Matplotlib, Seaborn**: Visualization of training and evaluation metrics.

## 📂 Dataset

- Source: [Kaggle - Brain Tumor Detection MRI Dataset](https://www.kaggle.com)
- Classes: 
  - `yes`: Tumor present (1500 images)
  - `no`: Tumor absent (1500 images)
  - `pred`: Test set (60 images)
- Images resized to `64x64`, grayscale.

## 🧪 Methodology

1. **Data Collection**: Kaggle MRI brain tumor dataset.
2. **Preprocessing**:
   - Resize to `64x64`
   - Normalize pixel values
   - One-hot encoding of labels
3. **Model Architecture**:
   - 3 Convolutional layers
   - Pooling layers
   - Fully connected dense layer
   - Dropout layer for regularization
4. **Training**:
   - Batch size: 16
   - Epochs: 10
   - Optimizer: Adam
5. **Evaluation**:
   - Accuracy: 98% (test and validation)
   - Confusion Matrix used for performance assessment

## 📈 Results

- **Training Accuracy**: 100%
- **Test Accuracy**: 98%
- **Validation Accuracy**: 98%
- The model shows strong generalization and performs well on unseen data.

## 🚀 Deployment

This project can be deployed as a web application using **Gradio** for real-time tumor detection from MRI image uploads (see code below).

## 📌 Future Work

- Add segmentation for tumor localization
- Expand dataset for better generalization
- Improve UI/UX for clinical use


