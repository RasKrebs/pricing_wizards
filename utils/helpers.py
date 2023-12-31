# Data Manipulation
import numpy as np
import time
from typing import Type
import pickle
import pandas as pd

# Scikit learn
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.utils import Bunch

# Utils
from utils.Dataloader import PricingWizardDataset

# PyTorch
import torch

def save_model(dictionary, path, model_type='sklearn') -> None:
    model = dictionary['model']

    # Save sklearn model
    if model is not None:
        if model_type == 'sklearn':
            with open(path, 'wb') as file:
                pickle.dump(model, file)
                print(f"Model saved successfully at {path}")
        elif model_type == 'pytorch':
            torch.save(model.state_dict(), path)
            print(f"Model saved successfully at {path}")
    else:
        print("No model found in the dictionary.")


def load_model(path):
    """Loads model from the given path"""
    with open(path, 'rb') as file:
        model = pickle.load(file)
        print(f"Model loaded successfully from {path}")
        return model

drop_helpers = lambda x: x.loc[:, (x.columns != 'classified_id') & (x.columns != 'listing_price') & (x.columns != 'log_listing_price')]

def two_step_hyperparameter_tuning(model: Type[BaseEstimator], dataset: PricingWizardDataset, param_dist: dict) -> Type[Bunch]:
    """
    Perform a two-step hyperparameter tuning process using Randomized Search and Grid Search.

    Parameters:
    - model (Type[BaseEstimator]): The machine learning model to be tuned.
    - dataset (PricingWizardDataset): The dataset containing training and testing data.
    - param_dist (dict): The hyperparameter distribution for Randomized Search.

    Returns:
    Type[Bunch]: A Bunch object containing the following attributes:
        - params: The best hyperparameters obtained from Grid Search.
        - model: The trained model with the best hyperparameters.
        - feature_importances: A list of tuples containing feature names and their importances.
        - training_time: The time taken to train the final model in seconds.
        - mse_mean_cv: The mean cross-validated mean squared error.
        - mse_test: The mean squared error on the test set.
        - y_pred: The predicted values on the test set.
    """

    X_train = drop_helpers(dataset.X_train).values
    y_train = dataset.y_train
    X_test = drop_helpers(dataset.X_test).values
    y_test = dataset.y_test

    # Check if the model is a SVM to apply StandardScaler
    if 'SVR' in str(type(model)) or 'LinearSVR' in str(type(model)):
        # Use StandardScaler for SVM models
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    # Use Random Search as first step of the hyperparameter tuning
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        scoring='neg_mean_squared_error',
        cv=3,
        n_iter=5,
        random_state=42,
        n_jobs=-1,
        verbose=3
    )
    random_search.fit(X_train, y_train)

    # Get the best hyperparameters from Random Search
    best_params_random: list = random_search.best_params_

    # Use the best hyperparameters from Random Search as initial values for Grid Search
    grid_search_params = {
        key: [value] for key, value in best_params_random.items()
    }

    grid_search = GridSearchCV(
        model,
        param_grid=grid_search_params,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=3
    )
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters from Grid Search
    best_params_grid: list = grid_search.best_params_

    # Train the final model with the best hyperparameters from Grid Search
    final_model = grid_search.best_estimator_

    # Evaluate the model using cross-validation and calculates the mean
    cv_scores: list = cross_val_score(final_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    mse_mean_cv: float = -np.mean(cv_scores)

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

    output: Type[Bunch] = Bunch(
        params = best_params_grid,
        model = final_model,
        feature_importances = list(zip(drop_helpers(dataset.X_train).columns, feature_importances)),
        training_time = training_time,
        mse_mean_cv = mse_mean_cv,
        mse_test = mse_test,
        y_pred = y_pred_test
    )

    return output

def set_feature_importances(results) -> None:
    """
    Create a pandas DataFrame from the 'feature_importances' key of the results dictionary and print it.

    Parameters:
    - results (dict): The dictionary containing the results, with a 'feature_importances' key.

    Returns:
    None
    """

    df = pd.DataFrame(results['feature_importances'][1:], columns=results['feature_importances'][0])

    print(df)
