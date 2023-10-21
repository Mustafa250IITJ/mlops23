"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm

from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations, tune_hparams
from joblib import dump, load
import pandas as pd
import pdb
import sys 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
num_runs = int(sys.argv[1])

def read_datasets():
    # Load your dataset here or replace this code with your actual data loading logic
    # For example, if you're using scikit-learn datasets:
    data = load_digits()
    X, y = data.data, data.target

    return X, y

# Load your dataset (replace this with your data loading code)
X, y = read_datasets()

# Define hyperparameter combinations for different models
# Example for SVM and Decision Trees:
svm_params = {
    'C': [0.1, 1, 10],
    'gamma': [0.0001, 0.001, 0.01]
}
tree_params = {
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create a dictionary of model parameters
model_params = {
    'svm': svm_params,
    'decision_tree': tree_params
}

# Define test sizes and dev sizes
test_sizes = [0.2]
dev_sizes = [0.2]

# Create a list to store results
results = []

# Loop over the number of runs
for cur_run_i in range(num_runs):
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1 - test_size - dev_size

            # Split the data into training, testing, and dev sets
            X_train, X_test, X_dev, y_train, y_test, y_dev = split_data(X, y, test_size, dev_size)

            # Preprocess the data (implement your preprocessing code)
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)

            # Loop over different model types
            for model_type, params in model_params.items():
                # Tune hyperparameters and get the best model
                best_model, best_accuracy = train_model(X_train, y_train, X_dev, y_dev, params, model_type)

                # Evaluate the model on training, dev, and test sets
                train_acc = predict_and_eval(best_model, X_train, y_train)
                dev_acc = predict_and_eval(best_model, X_dev, y_dev)
                test_acc = predict_and_eval(best_model, X_test, y_test)

                # Store the results
                cur_run_results = {
                    'model_type': model_type,
                    'run_index': cur_run_i,
                    'train_acc': train_acc,
                    'dev_acc': dev_acc,
                    'test_acc': test_acc
                }
                results.append(cur_run_results)

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Load your production and candidate models (replace with the correct paths)
production_model = load("path_to_production_model.joblib")
candidate_model = load("path_to_candidate_model.joblib")

# Make predictions using the production and candidate models and calculate metrics
production_predictions = production_model.predict(X_test)
candidate_predictions = candidate_model.predict(X_test)

production_accuracy = accuracy_score(y_test, production_predictions)
candidate_accuracy = accuracy_score(y_test, candidate_predictions)

# Calculate Confusion Matrix between Predictions of Production and Candidate Models (10x10 Matrix)
production_confusion = confusion_matrix(y_test, production_predictions)
candidate_confusion = confusion_matrix(y_test, candidate_predictions)
print("Production Model Confusion Matrix:")
print(production_confusion)
print("Candidate Model Confusion Matrix:")
print(candidate_confusion)

# Calculate Confusion Matrix for Correct Predictions in Production but Not in Candidate (2x2 Matrix)
correct_in_production_not_in_candidate = confusion_matrix(
    (production_predictions == y_test).astype(int),
    (candidate_predictions == y_test).astype(int)
)
print("Confusion Matrix for Correct in Production, Not in Candidate:")
print(correct_in_production_not_in_candidate)

# Calculate Macro-Average F1 Metrics
production_f1 = f1_score(y_test, production_predictions, average='macro')
candidate_f1 = f1_score(y_test, candidate_predictions, average='macro')
print("Production Model Macro-Average F1 Score:", production_f1)
print("Candidate Model Macro-Average F1 Score:", candidate_f1)

# print(results_df.groupby('model_type').describe().T)


# print(results_df[results_df['model_type'] == 'svm'].describe())
# print(results_df[results_df['model_type'] == 'decision_tree'].describe())


# pdb.set_trace()   # set breakpoint

# print ("lets check results df")