from sklearn.svm import SVR
from helpers.model_helpers import two_step_hyperparameter_tuning

def run_model(model_config):
    """
    Run a machine learning model using hyperparameter tuning with both GridSearchCV and RandomizedSearchCV.

    Parameters:
    - model_config (object): An object containing data and configurations for model training and testing.
        - X (array-like): The feature matrix for the entire dataset.
        - y (array-like): The target values for the entire dataset.
        - X_train (array-like): The feature matrix for the training dataset.
        - y_train (array-like): The target values for the training dataset.
        - X_test (array-like): The feature matrix for the test dataset.
        - y_test (array-like): The target values for the test dataset.

    Returns:
    - results (dict): A dictionary containing the results of hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
    """

    # Defines a set of values to explore during the hyperparameter tuning process
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'degree': [2, 3, 4]
    }

    # Create an SVR model
    svr = SVR()

    # Using param_grid for two step hyperparameter tuning with Support Vector Regression
    output = two_step_hyperparameter_tuning(svr, model_config, param_grid)

    return output
