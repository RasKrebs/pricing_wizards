# Data Manipulation
import os
import numpy as np
import time
import math
import pandas as pd
from typing import Type
from tabulate import tabulate
import pickle

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit learn
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score, mean_squared_log_error, median_absolute_error
from sklearn.utils import Bunch

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

def print_prediction_summary(label: str, y_true: pd.Series, y_pred: pd.Series) -> None:
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

def two_step_hyperparameter_tuning(model: Type[BaseEstimator],
                                   prediction_instance: Type["Prediction"],
                                   param_dist: dict) -> Type[Bunch]:
    """
    Perform two-step hyperparameter tuning using Random Search followed by Grid Search.

    Parameters:
    - model (BaseEstimator): The machine learning model class (e.g., SVR).
    - prediction_instance (Prediction): An object containing training and test data.
    - param_dist (dict): The hyperparameter grid to search.

    Returns:
    Bunch: A Bunch containing the results of the hyperparameter tuning
    """
    X_val = prediction_instance.X_val
    X_train = prediction_instance.X_train.values
    y_train = prediction_instance.y_train
    X_test = prediction_instance.X_test.values
    y_test = prediction_instance.y_test

    # Check if the model is a SVM to apply StandardScaler
    if isinstance(model, BaseEstimator) and 'SVR' in str(type(model)):
        # Use StandardScaler for SVM models
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    # Use Random Search as first step of the hyperparameter tuning
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        scoring='neg_mean_squared_error',
        cv=5,
        random_state=42
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
        cv=5
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
        feature_importances = list(zip(X_val.columns, feature_importances)),
        training_time = training_time,
        mse_mean_cv = mse_mean_cv,
        mse_test = mse_test,
        y_pred = y_pred_test
    )

    return output

def plot_actual_predicted(results: Type[Bunch], y_test: pd.Series) -> None:
    """
    Generate scatter plots for actual vs. predicted values for each model in the results.

    Parameters:
    - results (Type[Bunch]): A nested dictionary containing model results. Each model should have a 'y_pred' attribute
                            representing predicted values and a 'label' attribute for identification.
    - y_test (pd.Series): The actual values for the test set.

    Returns:
    None
    """

    models = [model for models in results.values() for model in models]

    num_models = len(models)
    num_cols = 2
    num_rows = math.ceil(num_models / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 6*num_rows), tight_layout=True)
    axs = axs.flatten()

    for i, model in enumerate(models):
        axs[i].scatter(y_test, model.y_pred, alpha=0.25)
        axs[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k', lw=1)
        axs[i].set_title(model.label)
        axs[i].set_xlabel('Actual Values')
        axs[i].set_ylabel('Predicted Values')

    plt.suptitle("Actual vs. Predicted values by model")

    save_plot(fig, "visualization/plot_actual_predicted")

def plot_residuals(results: Type[Bunch], y_test: pd.Series) -> None:
    """
    Generate scatter plots of residuals for each model in the results.

    Parameters:
    - results (Type[Bunch]): A nested dictionary containing model results. Each model should have a 'y_pred' attribute
                            representing predicted values and a 'label' attribute for identification.
    - y_test (pd.Series): The actual values for the test set.

    Returns:
    None
    """

    models = [model for models in results.values() for model in models]

    num_models = len(models)
    num_cols = 2
    num_rows = math.ceil(num_models / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 6*num_rows), tight_layout=True)

    for i, model in enumerate(models):
        # Calculate residuals
        prediction_error = y_test[i] - model.y_pred

        # Extract the subplot for the current model
        row = i // num_cols
        col = i % num_cols

        # Plot the scatter plot on the specific subplot
        axs[row, col].scatter(model.y_pred, prediction_error, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        axs[row, col].set_title(model.label)
        axs[row, col].set_xlabel('Predicted Values')
        axs[row, col].set_ylabel('Prediction Errors')

    plt.suptitle("Residual values by model")

    save_plot(fig, "visualization/plot_residuals")

def plot_model_evaluation(results: Type[Bunch]) -> None:
    regressor_names = [result.label for results in results.values() for result in results]

    mse_mean_cv_values = [result.mse_mean_cv for results in results.values() for result in results]
    mse_test_values = [result.mse_test for results in results.values() for result in results]

    bar_width = 0.35
    index = range(len(regressor_names))

    fig, ax = plt.subplots(tight_layout=True)
    bar1 = ax.bar(index, mse_mean_cv_values, bar_width, label='MSE Mean CV')
    bar2 = ax.bar([i + bar_width for i in index], mse_test_values, bar_width, label='MSE Test')

    ax.set_xlabel('Regressor')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title('MSE Mean CV and MSE Test for Each Regressor')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(regressor_names)
    ax.legend()

    save_plot(fig, "visualization/plot_model_evaluation")

def plot_feature_importances(feature_importances: pd.DataFrame) -> None:
    """
    Plot bar charts for average feature importances.

    Parameters:
    - ranked_feature_importances (pd.DataFrame): DataFrame with ranked feature importances.

    Returns:
    None
    """
    fig, ax = plt.subplots(tight_layout=True)

    # Bar plot for average feature importances
    sns.barplot(x=feature_importances.index, y=feature_importances['average'], hue=feature_importances.index, legend=False)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_title('Average Feature Importances')
    ax.set_xlabel('Features')
    ax.set_ylabel('Average Importance')
    ax.tick_params(axis='x', labelsize=8)

    save_plot(fig, "visualization/plot_feature_importances")

def plot_training_time(results: Type[Bunch]) -> None:
    regressor_names = [result.label for results in results.values() for result in results]
    training_times = [result.training_time for results in results.values() for result in results]

    fig, ax = plt.subplots(tight_layout=True)
    ax.bar(regressor_names, training_times, color='blue')
    ax.set_xlabel('Regressor')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Time Comparison for Different Regressors')

    save_plot(fig, "visualization/plot_training_time")

def save_plot(fig, path) -> None:
    if not os.path.exists("visualization"):
        os.makedirs("visualization")

    fig.savefig(path)
