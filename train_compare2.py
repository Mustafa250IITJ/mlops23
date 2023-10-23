import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Step 1: Import Required Libraries

# Step 2: Load the MNIST Dataset
mnist = datasets.load_digits()
X, y = mnist.data, mnist.target

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Production Model
production_model = SVC(C=1.0, kernel='rbf')  # Example hyperparameters, perform tuning
production_model.fit(X_train, y_train)

# Step 5: Train the Candidate Model
candidate_model = DecisionTreeClassifier(max_depth=None)  # Example hyperparameters, perform tuning
candidate_model.fit(X_train, y_train)

# Step 6: Evaluate Model Accuracy
production_accuracy = accuracy_score(y_test, production_model.predict(X_test))
candidate_accuracy = accuracy_score(y_test, candidate_model.predict(X_test))

# Step 7: Calculate Confusion Matrix
production_confusion_matrix = confusion_matrix(y_test, production_model.predict(X_test))
candidate_confusion_matrix = confusion_matrix(y_test, candidate_model.predict(X_test))

# Step 8: Calculate 2x2 Confusion Matrix
correct_in_production_not_in_candidate = np.sum((production_model.predict(X_test) == y_test) & (candidate_model.predict(X_test) != y_test))
confusion_matrix_2x2 = np.array([[correct_in_production_not_in_candidate, 0],[0, 0]])
                                 

# Step 9: Report Macro-Average F1 Score
production_f1 = f1_score(y_test, production_model.predict(X_test), average='macro')
candidate_f1 = f1_score(y_test, candidate_model.predict(X_test), average='macro')

# Step 10: Display Results
print("Production Model Accuracy:", production_accuracy)
print("Candidate Model Accuracy:", candidate_accuracy)
print("Production Model Confusion Matrix:\n", production_confusion_matrix)
print("Candidate Model Confusion Matrix:\n", candidate_confusion_matrix)
print("2x2 Confusion Matrix (Correct in Production but Not in Candidate):\n", confusion_matrix_2x2)
print("Production Model Macro-Average F1 Score:", production_f1)
print("Candidate Model Macro-Average F1 Score:", candidate_f1)
