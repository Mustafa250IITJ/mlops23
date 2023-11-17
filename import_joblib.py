from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
# Folder where the trained machine learning model is saved
model_folder = 'trainmodelsaved/'
# model_folder = '/app/trainmodelsaved'
# Construct the full path to the trained model
model_path = os.path.join(model_folder, 'mnist_model.joblib')
# Load the trained machine learning model
loaded_model = joblib.load(model_path)

# Access the parameters/attributes of the model
if isinstance(loaded_model, LogisticRegression):
    # Assuming the model is a Logistic Regression model
    coefficients = loaded_model.coef_
    intercept = loaded_model.intercept_
    
    print("Coefficients:", coefficients)
    print("Intercept:", intercept)
else:
    print("Model type not recognized. Check the model type and access attributes accordingly.")

# print("length of coefficients:", len(coefficients))
# print("length of Intercept:",len(intercept))

# Load the MNIST dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Make predictions on the test set
y_pred = loaded_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")