from typing import Type
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import Bunch

from utils.object import MLModelConfig
from utils.prediction import two_step_hyperparameter_tuning
from utils.prediction import print_prediction_summary

def main(model_config: MLModelConfig) -> Type[Bunch]:
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
    - results (Type[Bund]): A dictionary containing the results of hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
    """

    # Defines a set of values to explore during the hyperparameter tuning process
    param_grid: dict = {
        'preprocessor__cat__handle_unknown': ['ignore'],
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

    # Create a Random Forest model
    rf = RandomForestRegressor()

    # Using param_grid for two step hyperparameter tuning with Random Forest
    output_rf: Type[Bunch] = two_step_hyperparameter_tuning(rf, model_config, param_grid)

    # Add label to output
    output_rf.label = 'Random Forest'

    output: Type[Bunch] = Bunch(
        standard = output_rf
    )

    # Printing a summary of the results
    print_prediction_summary('Random Forest', model_config.y_test, output_rf.y_pred)

    return output
