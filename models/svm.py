# Data Manipulation
from typing import Type

# Scikit learn
from sklearn.svm import SVR
from sklearn.utils import Bunch

# Load helpers and custom dataset class
from utils.Dataloader import PricingWizardDataset
from utils.RegressionEvaluation import regression_accuracy
from utils.helpers import two_step_hyperparameter_tuning

def svm(dataset: PricingWizardDataset):
    # Defines a set of values to explore during the hyperparameter tuning process
    param_dist: dict = {
        'C': [1, 10, 100, 1000],
        'gamma': [0.1, 0.01, 0.001, 0.0001],
    }

    # Create an SVR linear model
    svr_linear = SVR(kernel="linear")

    # Create an SVR linear model
    svr_rbf = SVR(kernel="rbf")

    # Create an SVR linear model
    svr_poly = SVR(kernel="poly")

    # Using param_grid for two step hyperparameter tuning with Support Vector Regression
    output_linear: Type[Bunch] = two_step_hyperparameter_tuning(svr_linear, dataset, param_dist)
    output_rbf: Type[Bunch] = two_step_hyperparameter_tuning(svr_rbf, dataset, param_dist)
    output_poly: Type[Bunch] = two_step_hyperparameter_tuning(svr_poly, dataset, param_dist)

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