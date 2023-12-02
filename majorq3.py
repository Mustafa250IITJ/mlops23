import joblib
import pytest
from sklearn.linear_model import LogisticRegression

roll_no = 'm22aie250'

@pytest.mark.parametrize("solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
def test_model_type_and_solver(solver):
    # Load the model
    model_name = f'{roll_no}_lr_{solver}.joblib'
    loaded_model = joblib.load(model_name)

    # Check that the loaded model is an instance of LogisticRegression
    assert isinstance(loaded_model, LogisticRegression), f"Model {model_name} is not a Logistic Regression model."

    # Check that the solver in the model file name matches the solver used in the model
    assert solver == loaded_model.get_params()['solver'], f"Solver mismatch for model {model_name}. Expected: {solver}, Actual: {loaded_model.get_params()['solver']}"
