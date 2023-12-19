# Data Manipulation
from typing import Type

# Scikit learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import Bunch

# Load helpers and custom dataset class
from utils.Dataloader import PricingWizardDataset
from utils.RegressionEvaluation import regression_accuracy
from utils.helpers import two_step_hyperparameter_tuning

def random_forest(dataset: PricingWizardDataset) -> Type[Bunch]:
    """
    Perform hyperparameter tuning and evaluation for a Random Forest regression model.

    Parameters:
    - dataset (PricingWizardDataset): The dataset containing training and testing data.

    Returns:
    Type[Bunch]: A Bunch object containing the Random Forest model with hyperparameter tuning results and evaluation metrics.
    The Bunch object has the following attributes:
        - model: Trained Random Forest model with the best hyperparameters.
        - r2: R-squared.
        - mse: Mean squared error.
        - mae: Mean absolute error.
        - rmse: Root mean squared error.
        - training_time: Time taken to train the final model in seconds.
        - mse_mean_cv: Mean cross-validated mean squared error.
        - mse_test: Mean squared error on the test set.
        - feature_importances: List of tuples containing feature names and their importances.
    """

    # Defines a set of values to explore during the hyperparameter tuning process
    param_dist: dict = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a Random Forest model
    rf = RandomForestRegressor()

    # Using param_dist for two step hyperparameter tuning with Random Forest
    output: Type[Bunch] = two_step_hyperparameter_tuning(rf, dataset, param_dist)

    r2, mse, mae, rmse = regression_accuracy(output.y_pred, dataset.y_test, return_metrics=True)

    # Create a dictionary to store results and model
    results = {
        'model': output.model,
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'training_time': output.training_time,
        'mse_mean_cv': output.mse_mean_cv,
        'mse_test': output.mse_test,
        'feature_importances': output.feature_importances
    }

    return results