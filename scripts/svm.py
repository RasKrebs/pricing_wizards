from __future__ import annotations

# Data Manipulation
from typing import Type

# Scikit learn
from sklearn.svm import SVR
from sklearn.utils import Bunch

# Load helpers and custom dataset class
from utils.helpers import two_step_hyperparameter_tuning, print_prediction_summary, save_model

def run_svm(prediction_instance: Type["Prediction"]) -> Type[Bunch]:
    """
    Run a machine learning model using hyperparameter tuning with both GridSearchCV and RandomizedSearchCV.

    Parameters:
    - prediction_instance (Prediction): An object containing data and configurations for model training and testing.

    Returns:
    - results (Type[Bunch]): A dictionary containing the results of hyperparameter tuning using
                      GridSearchCV and RandomizedSearchCV for SVR Linear, RBF and Polynomial.
    """

    # Defines a set of values to explore during the hyperparameter tuning process
    param_dist: dict = {
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
    output_linear: Type[Bunch] = two_step_hyperparameter_tuning(svr_linear, prediction_instance, param_dist)
    output_rbf: Type[Bunch] = two_step_hyperparameter_tuning(svr_rbf, prediction_instance, param_dist)
    output_poly: Type[Bunch] = two_step_hyperparameter_tuning(svr_poly, prediction_instance, param_dist)

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

    for label, model in output_svm.items():
        path = f'models/pickled_models/prediction_{"_".join(model.label.lower().split())}.pkl'
        # Saving model
        save_model(model, path)

        # Printing a summary of the results
        print_prediction_summary(f"SVM {label}", prediction_instance.y_test, model.y_pred)

    return output_svm