import os
import cv2
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# ==============================
# SETTINGS
# ==============================
DATADIR = "PetImages"
CATEGORIES = ["Cat", "Dog"]
IMG_SIZE = 224   # ✅ Better for VGG16
LIMIT_PER_CLASS = 2000

data = []

# ==============================
# LOAD DATA
# ==============================
def create_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        count = 0
        for img in tqdm(os.listdir(path), desc=f"Loading {category}"):

            if count >= LIMIT_PER_CLASS:
                break

            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

                if image.shape != (IMG_SIZE, IMG_SIZE, 3):
                    continue

                data.append([image, class_num])
                count += 1

            except:
                continue

create_data()

print("Total images:", len(data))

# ==============================
# PREPARE DATA
# ==============================
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X, dtype="float32")
y = np.array(y)

# ✅ Correct preprocessing for VGG16
X = preprocess_input(X)

# ==============================
# CNN FEATURE EXTRACTION
# ==============================
print("\n⏳ Extracting CNN features...")

base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))

# ✅ Freeze layers (faster)
for layer in base_model.layers:
    layer.trainable = False

# ✅ Better feature extraction
x = base_model.output
x = GlobalAveragePooling2D()(x)

model = Model(inputs=base_model.input, outputs=x)

# Extract features
features = model.predict(X, batch_size=32, verbose=1)

X_features = features

print("Feature shape:", X_features.shape)

# ==============================
# SHUFFLE
# ==============================
X_features, y = shuffle(X_features, y, random_state=42)

# ==============================
# TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# FEATURE SCALING (IMPORTANT)
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# TRAIN SVM
# ==============================
print("\n⏳ Training SVM...")

svm = SVC(kernel='rbf', C=50, gamma=0.0005)

svm.fit(X_train, y_train)

print("✅ Training completed")

# ==============================
# EVALUATION
# ==============================
y_pred = svm.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))