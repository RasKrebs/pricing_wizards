from typing import Type
from sklearn.svm import SVR
from utils.prediction import two_step_hyperparameter_tuning
from sklearn.utils import Bunch

from utils.object import MLModelConfig
from utils.prediction import two_step_hyperparameter_tuning
from utils.prediction import print_prediction_summary

def main(model_config: MLModelConfig) -> Type[Bunch]:
    """
    Run a machine learning model using hyperparameter tuning with both GridSearchCV and RandomizedSearchCV.

    Parameters:
    - model_config (MLModelConfig): An object containing data and configurations for model training and testing.
        - X (array-like): The feature matrix for the entire dataset.
        - y (array-like): The target values for the entire dataset.
        - X_train (array-like): The feature matrix for the training dataset.
        - y_train (array-like): The target values for the training dataset.
        - X_test (array-like): The feature matrix for the test dataset.
        - y_test (array-like): The target values for the test dataset.

    Returns:
    - results (Type[Bunch]): A dictionary containing the results of hyperparameter tuning using
                      GridSearchCV and RandomizedSearchCV for SVR Linear, RBF and Polynomial.
    """

    # Defines a set of values to explore during the hyperparameter tuning process
    param_grid: dict = {
        'preprocessor__cat__handle_unknown': ['ignore'],
        'regressor__C': [0.1, 1, 10],
        'regressor__gamma': [0.01, 0.1, 1],
        'regressor__degree': [2, 3, 4]
    }

    # Create an SVR linear model
    svr_linear = SVR(kernel="linear")

    # Create an SVR linear model
    svr_rbf = SVR(kernel="rbf")

    # Create an SVR linear model
    svr_poly = SVR(kernel="poly")

    # Using param_grid for two step hyperparameter tuning with Support Vector Regression
    output_linear: Type[Bunch] = two_step_hyperparameter_tuning(svr_linear, model_config, param_grid)
    output_rbf: Type[Bunch] = two_step_hyperparameter_tuning(svr_rbf, model_config, param_grid)
    output_poly: Type[Bunch] = two_step_hyperparameter_tuning(svr_poly, model_config, param_grid)

    # Add labels to outputs
    output_linear.label = 'SVR Linear'
    output_rbf.label = 'SVR RBF'
    output_poly.label = 'SVR Polynomial'

    # Generate output with model label
    output_svm: Type[Bunch] = Bunch(
        linear = output_linear,
        rbf = output_rbf,
        polynomial = output_poly
    )

    # Printing a summary of the results
    for label, regressor in output_svm.items():
        print_prediction_summary(f"SVM {label}", model_config.y_test, regressor.y_pred)

    return output_svm
