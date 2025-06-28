# Dog-Cat-Classifier-Svm 

"company": SkillCraft Technology

"NAME":DIVYANSH GAHLAUT

"INTERN ID" :SCT/JUN25/6158

"DOMAIN" :  Machine Learning

"DURATION": 4 WEEKS

*Project Description:

This project aims to build a machine learning image classification system using a Support Vector Machine (SVM) to differentiate between images of cats and dogs. Instead of deep learning, it leverages traditional ML approaches (SVM) to solve a supervised classification problem on visual data.

The model is trained using the PetImages dataset available from Kaggle, containing thousands of labeled images of dogs and cats. To reduce computation time and focus on core ML principles, the dataset is limited to a smaller subset (e.g., 1000 images per class).

This project is part of SkillCraft Technology Task 03 and is implemented using Python and scikit-learn, showcasing the practical use of SVMs for computer vision tasks in a resource-efficient way

🧰 Tools & Technologies Used:

1.Python 🐍

2.OpenCV (cv2) – for image loading and preprocessing

3.NumPy – for array and matrix manipulation

4.scikit-learn (sklearn) – for model training, evaluation, and prediction

5.matplotlib – for image visualization (optional, for prediction demo)

6.joblib – to save and load the trained model

7.Jupyter Notebook / VS Code – development environment

⚙️ How It Works:

🔹 1. Dataset Loading & Preprocessing
Images are sourced from the PetImages folder downloaded from Kaggle.

Each image is:

Loaded using OpenCV

Converted to grayscale to reduce complexity

Resized to a fixed dimension (e.g., 100×100 pixels)

Flattened into a 1D vector (100×100 = 10,000 features)

🔹 2. Labeling
Each image is assigned a label:

0 for "Dog"

1 for "Cat"

Data is appended to two lists: data (image vectors) and labels (corresponding targets).

🔹 3. Splitting Data
The dataset is split into training (80%) and testing (20%) using train_test_split().

🔹 4. Model Training
A Support Vector Classifier (SVC) with a linear kernel is trained using scikit-learn.

Training accuracy and performance are measured.

The model is saved as svm_grayscale_model.pkl.

🔹 5. Evaluation
The trained model is evaluated on the test set.

Metrics used:

Accuracy

Precision

Recall

F1-Score

Results are printed in a classification report.

🔹 6. Prediction on New Image
A test image is read, resized, and reshaped similarly.

The trained model predicts whether the image is a cat or dog.

Optional: the image is displayed using matplotlib, and the prediction is printed.

#OUTPUT:

![Image](https://github.com/user-attachments/assets/c2d7aa0d-cabc-4342-a146-2a856a1cc91e)

![Image](https://github.com/user-attachments/assets/bfa4a10d-7f2d-4efa-abc1-ac396f31b5ec)


![Image](https://github.com/user-attachments/assets/e7ecc1f3-afa8-4848-8c15-b708645ea708)





