import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ==== Configuration ====
DATADIR = "PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 100
MODEL_PATH = "svm_grayscale_model.pkl"

# ==== Load & Preprocess Images ====
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    count = 0

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(resized.flatten())
            labels.append(class_num)
            count += 1
            if count == 1000:  # Load only 1000 images per class
                break
        except Exception as e:
            continue

# ==== Prepare Data ====
data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# ==== Train Model ====
print("[INFO] Training SVM classifier...")
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# ==== Save the model ====
joblib.dump(model, MODEL_PATH)
print(f"[INFO] Model saved as {MODEL_PATH}")

# ==== Evaluate ====
print("[INFO] Evaluating model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred, target_names=CATEGORIES))
