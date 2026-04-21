# 🐶🐱 Cat vs Dog Classification using CNN + SVM

🚀 A Machine Learning project that combines **Deep Learning (VGG16 CNN)** and **Support Vector Machine (SVM)** to classify images of cats and dogs with high accuracy.

---

## 📌 Project Overview

This project uses a **pretrained VGG16 Convolutional Neural Network** as a **feature extractor**, followed by an **SVM classifier** to perform image classification.

Instead of training a full CNN from scratch, we leverage **transfer learning** for efficient and accurate predictions.

---

## 🎯 Objectives

* Extract deep features using **VGG16**
* Train a powerful **SVM classifier**
* Achieve high accuracy on image classification
* Build an efficient hybrid ML model

---

## 🧠 Tech Stack

* 🐍 Python
* 🤖 TensorFlow / Keras
* 📊 Scikit-learn
* 📸 OpenCV
* 📈 NumPy, Pandas
* 🔄 tqdm

---

## 📂 Dataset

* Dataset: **PetImages (Cats vs Dogs)**
* Classes: `Cat` 🐱 and `Dog` 🐶
* Images resized to **224x224**

> Make sure your dataset structure looks like:

```
PetImages/
    ├── Cat/
    └── Dog/
```

---

## ⚙️ Workflow

1. 📥 Load and preprocess images
2. 🧹 Clean corrupted images
3. 🔄 Resize & normalize data
4. 🧠 Extract features using VGG16
5. 📊 Flatten features
6. ⚖️ Train-test split
7. 📈 Train SVM classifier
8. 🎯 Evaluate model performance

---

## 🚀 How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/cnn-svm-cat-dog.git
cd cnn-svm-cat-dog
```

### 2️⃣ Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install tensorflow==2.15.0 scikit-learn numpy pandas matplotlib opencv-python tqdm
```

### 4️⃣ Run the project

```bash
python cnn_svm.py
```

---

## 📊 Results

* 🎯 Accuracy: **~90% - 96%**
* 📈 Balanced performance across both classes
* ⚡ Efficient training using transfer learning

---

## 🖼️ Sample Output

```
Accuracy: 0.93

Classification Report:

              precision    recall  f1-score   support

       Cat       0.92      0.94      0.93
       Dog       0.94      0.91      0.92
```

---

## 💡 Key Highlights

✔ Hybrid Model (CNN + SVM)
✔ Transfer Learning (VGG16)
✔ Optimized Feature Extraction
✔ Scalable & Efficient Pipeline
✔ Beginner-Friendly + Internship Ready

---

## 🔥 Future Improvements

* 📸 Real-time webcam classification
* ⚙️ Hyperparameter tuning (GridSearchCV)
* 📊 Confusion matrix visualization
* 🧠 Try advanced models (ResNet, EfficientNet)

---

## 🏢 Internship

This project was completed as part of my **Machine Learning Internship at Prodigy InfoTech**.

---

## 🙌 Connect with Me

* 💼 LinkedIn: https://linkedin.com/in/yashtambeml
* 💻 GitHub: https://github.com/yashtambeml

---

⭐ If you found this project useful, consider giving it a star!
