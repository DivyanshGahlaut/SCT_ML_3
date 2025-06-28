import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ==== Configuration ====
IMG_SIZE = 100
CATEGORIES = ["Dog", "Cat"]
MODEL_PATH = "../svm_grayscale_model.pkl"
IMAGE_PATH = os.path.join("..", "PetImages", "Cat", "2.jpg")


# ==== Load trained model ====
print("[INFO] Loading model...")
model = joblib.load(MODEL_PATH)

# ==== Load and preprocess image ====
print(f"[INFO] Loading image from {IMAGE_PATH}")
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"[ERROR] Could not load image from: {IMAGE_PATH}")
    exit()

resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
flattened = resized.flatten().reshape(1, -1)

# ==== Predict ====
print("[INFO] Predicting...")
prediction = model.predict(flattened)[0]
label = CATEGORIES[prediction]

# ==== Display image and result ====
plt.imshow(img, cmap='gray')
plt.title(f"Prediction: {label}")
plt.axis("off")
plt.show()

print(f"The predicted image is: {label}")
