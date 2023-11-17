from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the MNIST dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model (replace with your preferred model and hyperparameter tuning)
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


model_dir = 'trainmodelsaved'
os.makedirs(model_dir, exist_ok=True)
# Specify the model file name with the folder path
model_filename = os.path.join(model_dir, 'mnist_model.joblib')
# Save the Model
joblib.dump(model, model_filename)


# # Save the trained model to a file
# joblib.dump(model, 'mnist_model.joblib')
