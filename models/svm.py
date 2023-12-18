# Data Manipulation
from typing import Type

# Scikit learn
from sklearn.svm import SVR, LinearSVR
from sklearn.utils import Bunch

# Load helpers and custom dataset class
from utils.Dataloader import PricingWizardDataset
from utils.RegressionEvaluation import regression_accuracy
from utils.helpers import two_step_hyperparameter_tuning

def svm(dataset: PricingWizardDataset) -> Type[Bunch]:
    """
    Perform hyperparameter tuning and evaluation for Support Vector Regression (SVR) models.

    Parameters:
    - dataset (PricingWizardDataset): The dataset containing training and testing data.

    Returns:
    Type[Bunch]: A Bunch object containing SVR models with hyperparameter tuning results and evaluation metrics.
    The Bunch object has the following attributes:
        - linear: SVR with linear kernel results and evaluation metrics.
        - rbf: SVR with radial basis function (RBF) kernel results and evaluation metrics.
        - polynomial: SVR with polynomial kernel results and evaluation metrics.

    """

    # Defines a set of values to explore during the hyperparameter tuning process
    linear_param_dist = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5, 1.0],
    }

    poly_param_dist = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.5],
        'degree': [2, 3, 4],
    }

    rbf_param_dist = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.5],
        'gamma': ['scale', 'auto', 0.1],
    }

    # Create an SVR linear model
    svr_linear = LinearSVR(dual='auto')

    # Create an SVR linear model
    svr_rbf = SVR(kernel="rbf")

    # Create an SVR linear model
    svr_poly = SVR(kernel="poly")

    # Using param_grid for two step hyperparameter tuning with Support Vector Regression
    output_linear: Type[Bunch] = two_step_hyperparameter_tuning(svr_linear, dataset, linear_param_dist)
    output_rbf: Type[Bunch] = two_step_hyperparameter_tuning(svr_rbf, dataset, rbf_param_dist)
    output_poly: Type[Bunch] = two_step_hyperparameter_tuning(svr_poly, dataset, poly_param_dist)

    # Add labels to outputs
    output_linear.label = 'SVR Linear'
    output_rbf.label = 'SVR RBF'
    output_poly.label = 'SVR Polynomial'

    # Generate output with model label
    results: Type[Bunch] = Bunch(
        linear = output_linear,
        rbf = output_rbf,
        polynomial = output_poly
    )

    for _, params in results.items():
        # Calculate metrics
        r2, mse, mae, rmse = regression_accuracy(params.y_pred, dataset.y_test, return_metrics=True)

        # Create a dictionary to store results and model
        params.accuracy = {
            'model': params.model,
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'training_time': params.training_time,
            'mse_mean_cv': params.mse_mean_cv,
            'mse_test': params.mse_test,
            'feature_importances': params.feature_importances
        }

    return results