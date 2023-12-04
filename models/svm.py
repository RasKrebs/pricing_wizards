from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVR

from helpers.model_helpers import hyperparameter_tuning

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
        - 'GridSearch': A dictionary with the hyperparameter tuning results using GridSearchCV, as returned by hyperparameter_tuning.
        - 'RandomSearch': A dictionary with the hyperparameter tuning results using RandomizedSearchCV, as returned by hyperparameter_tuning.
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

    # Using GridSearchCV for hyperparameter tuning with Support Vector Regression
    grid_search = GridSearchCV(svr, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search_output = hyperparameter_tuning(SVR, model_config, grid_search)

    # Using RandomizedSearchCV for hyperparameter tuning with Support Vector Regression
    random_search = RandomizedSearchCV(svr, param_distributions=param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=5, random_state=42)
    random_search_output = hyperparameter_tuning(SVR, model_config, random_search)

    # Storing the results of hyperparameter tuning in a dictionary
    results = {
        'GridSearch': grid_search_output,
        'RandomSearch': random_search_output,
    }

    return results
