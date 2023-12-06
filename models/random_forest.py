from sklearn.ensemble import RandomForestRegressor
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
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a Random Forest model
    rf = RandomForestRegressor()

    # Using param_grid for two step hyperparameter tuning with Random Forest
    output = two_step_hyperparameter_tuning(rf, model_config, param_grid)

    # Get feature importances for Random Forest
    feature_importances = output['model'].feature_importances_

    # Add feature importances to output
    output['feature_importances'] = feature_importances

    return output
