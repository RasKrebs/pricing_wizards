import numpy as np
import time
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score, mean_squared_log_error, median_absolute_error
from sklearn.utils import Bunch
from typing import Type
from tabulate import tabulate

from utils.object import MLModelConfig

def two_step_hyperparameter_tuning(model_class: Type[BaseEstimator], model_config: MLModelConfig, param_grid: dict) -> Type[Bunch]:
    """
    Perform two-step hyperparameter tuning using Random Search followed by Grid Search.

    Parameters:
    - model_class (BaseEstimator): The machine learning model class (e.g., SVR).
    - model_config (MLModelConfig): An object containing training and test data (X_train, y_train, X_test, y_test).
    - param_grid (dict): The hyperparameter grid to search.

    Returns:
    Bunch: A Bunch containing the results of the hyperparameter tuning
    """

    X = model_config.X
    X_train = model_config.X_train
    y_train = model_config.y_train
    X_test = model_config.X_test
    y_test = model_config.y_test

    # Define categorical features
    categorical_features = X.columns.tolist()

    # Create a column transformer for one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Create a pipeline with named steps
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model_class)
    ])

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring='neg_mean_squared_error',
        cv=5,
        random_state=42
    )

    # Use Random Search as first step of the hyperparameter tuning
    random_search.fit(X_train, y_train)

    # Get the best hyperparameters from Random Search
    best_params_random: list = random_search.best_params_

    # Use the best hyperparameters from Random Search as initial values for Grid Search
    grid_search_params = {
        key: [value] for key, value in best_params_random.items()
    }

    grid_search = GridSearchCV(
        pipeline,
        grid_search_params,
        scoring='neg_mean_squared_error',
        cv=5
    )
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters from Grid Search
    best_params_grid: list = grid_search.best_params_

    # Train the final model with the best hyperparameters from Grid Search
    final_model = grid_search.best_estimator_

    # Evaluate the model using cross-validation and calculates the mean
    cv_scores: list = cross_val_score(final_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    mse_mean_cv: float = np.mean(cv_scores)

    # Train the final model on the entire training set, measuring the training time in seconds
    start_time = time.time()
    final_model.fit(X_train, y_train)
    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time

    # Evaluate the final model on the test set
    y_pred_test = final_model.predict(X_test)
    mse_test: float = mean_squared_error(y_test, y_pred_test)

    # Calculate permutation importances for the regressor
    feature_importances = permutation_importance(final_model, X_test, y_test, n_repeats=10, random_state=42).importances_mean

    output: type[Bunch] = Bunch(
        params = best_params_grid,
        model = final_model,
        feature_importances = list(zip(X.columns, feature_importances)),
        training_time = training_time,
        mse_mean_cv = mse_mean_cv,
        mse_test = mse_test,
        y_pred = y_pred_test
    )

    return output

def print_prediction_summary(label: str, y_true: list, y_pred: list) -> None:
    """
    Print a summary of regression evaluation metrics.

    Parameters:
    - y_true (list): True values of the target variable.
    - y_pred (list): Predicted values of the target variable.

    Returns:
    None
    """

    evs = round(explained_variance_score(y_true, y_pred), 4)
    msle = round(mean_squared_log_error(y_true, y_pred), 4)
    r2 = round(r2_score(y_true, y_pred), 4)
    mae = round(mean_absolute_error(y_true, y_pred), 4)
    mse = round(mean_squared_error(y_true, y_pred), 4)
    medae = round(median_absolute_error(y_true, y_pred), 4)
    rmse = round(np.sqrt(mse), 4)

    table = [
        ["Explained Variance", evs],
        ["Mean Squared Log Error", msle],
        ["R2", r2],
        ["MAE", mae],
        ["MSE", mse],
        ["Median Absolute Error", medae],
        ["Root Mean Squared Error", rmse]
    ]

    print(tabulate(table, headers=[f"Metric ({label})", "Value"], tablefmt="pretty", numalign="right", stralign="right", colalign=("left", "right")))
