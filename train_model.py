import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import joblib  # Import joblib to save the model
import os

# Load the MNIST Dataset
mnist = datasets.load_digits()
X, y = mnist.data, mnist.target

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Production Model
production_model = SVC(C=1.0, kernel='rbf')  # hyperparameters tuning
production_model.fit(X_train, y_train)

# Train the Candidate Model
candidate_model = DecisionTreeClassifier(max_depth=None)  # hyperparameters tuning
candidate_model.fit(X_train, y_train)

# # Save the Candidate Model
# model_filename = 'candidate_model.pkl'  # Specify the model file name
# joblib.dump(candidate_model, model_filename)

# Create a 'model' directory if it doesn't exist
model_dir = 'train_modelsaved'
os.makedirs(model_dir, exist_ok=True)
# Specify the model file name with the folder path
model_filename = os.path.join(model_dir, 'dts_model.joblib')
# Save the Model
joblib.dump(candidate_model, model_filename)


# Evaluate Model Accuracy
production_accuracy = accuracy_score(y_test, production_model.predict(X_test))
candidate_accuracy = accuracy_score(y_test, candidate_model.predict(X_test))

# Calculate Confusion Matrix
production_confusion_matrix = confusion_matrix(y_test, production_model.predict(X_test))
candidate_confusion_matrix = confusion_matrix(y_test, candidate_model.predict(X_test))

# Calculate 2x2 Confusion Matrix
correct_in_production_not_in_candidate = np.sum((production_model.predict(X_test) == y_test) & (candidate_model.predict(X_test) != y_test))
confusion_matrix_2x2 = np.array([[correct_in_production_not_in_candidate, 0], [0, 0]])

# Report Macro-Average F1 Score
production_f1 = f1_score(y_test, production_model.predict(X_test), average='macro')
candidate_f1 = f1_score(y_test, candidate_model.predict(X_test), average='macro')

# Display Results
# print("Production Model Accuracy:", production_accuracy)
# print("Candidate Model Accuracy:", candidate_accuracy)
# print("Production Model Confusion Matrix:\n", production_confusion_matrix)
# print("Candidate Model Confusion Matrix:\n", candidate_confusion_matrix)
# print("2x2 Confusion Matrix (Correct in Production but Not in Candidate):\n", confusion_matrix_2x2)
# print("Production Model Macro-Average F1 Score:", production_f1)
# print("Candidate Model Macro-Average F1 Score:", candidate_f1)

print("model saved as:", model_filename)  # Print the saved model's filename
