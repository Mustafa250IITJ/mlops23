from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
import joblib
from joblib import dump, load

# Load the Digits dataset
digits = load_digits()
data = digits.data
target = digits.target

# Set up Logistic Regression model
lr_model = LogisticRegression()

# Define the hyperparameters to search
param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

# Perform GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(data, target)

# Report mean and std across 5 CV for each solver
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']

roll_no = 'm22aie250'

# Save models for each solver
for mean, std, solver_params in zip(means, stds, params):
    solver = solver_params['solver']
    
    # Set the best hyperparameters to the model
    lr_model.set_params(**solver_params)
    
    # Fit the model to the entire dataset
    lr_model.fit(data, target)
    
    # Save the model
    model_name = f'{roll_no}_lr_{solver}.joblib'
    joblib.dump(lr_model, model_name)

    print(f'Model saved as {model_name}')

# Print the best hyperparameters and corresponding accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(f'Best Hyperparameters: {best_params}')
print(f'Best Accuracy: {best_accuracy:.4f}')