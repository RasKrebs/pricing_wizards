from sklearn.svm import SVR
from helpers.model_helpers import two_step_hyperparameter_tuning
from sklearn.inspection import permutation_importance

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
    - results (dict): A dictionary containing the results of hyperparameter tuning using
                      GridSearchCV and RandomizedSearchCV for SVR Linear, RBF and Polynomial.
    """

    # Defines a set of values to explore during the hyperparameter tuning process
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'degree': [2, 3, 4]
    }

    # Create an SVR linear model
    svr_linear = SVR(kernel="linear")

    # Create an SVR linear model
    svr_rbf = SVR(kernel="rbf")

    # Create an SVR linear model
    svr_poly = SVR(kernel="poly")

    # Using param_grid for two step hyperparameter tuning with Support Vector Regression
    output_linear = two_step_hyperparameter_tuning(svr_linear, model_config, param_grid)
    output_rbf = two_step_hyperparameter_tuning(svr_rbf, model_config, param_grid)
    output_poly = two_step_hyperparameter_tuning(svr_poly, model_config, param_grid)

     # Get coefficients for linear kernel SVR
    coefficients = output_linear['model'].coef_

    # Calculate feature importances with permutation importance
    feature_importances = permutation_importance(output_linear['model'], model_config.X, model_config.y, scoring='neg_mean_squared_error')

    # Add feature importances to output
    output_linear['feature_importances'] = feature_importances.importances

    # Generate output
    output = {
        'SVR Linear': output_linear,
        'SVR RBF': output_rbf,
        'SVR Polynomial': output_poly,
    }

    return output
