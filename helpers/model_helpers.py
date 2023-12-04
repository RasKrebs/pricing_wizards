from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

def hyperparameter_tuning(model_class, model_config, optimizer):
    """
    Perform hyperparameter tuning for a given machine learning model using the specified optimizer.

    Parameters:
    - model_class (class): The machine learning model class (e.g., sklearn.ensemble.RandomForestRegressor).
    - model_config (object): An object containing data and configurations for model training and testing.
        - X (array-like): The feature matrix for the entire dataset.
        - y (array-like): The target values for the entire dataset.
        - X_train (array-like): The feature matrix for the training dataset.
        - y_train (array-like): The target values for the training dataset.
        - X_test (array-like): The feature matrix for the test dataset.
        - y_test (array-like): The target values for the test dataset.
    - optimizer (object): The hyperparameter optimization algorithm or tool (e.g., GridSearchCV).

    Returns:
    - output (dict): A dictionary containing the following information:
        - 'params': The best hyperparameters determined by the optimizer.
        - 'model': The trained model with the best hyperparameters.
        - 'mse_mean_cv': The mean cross-validated mean squared error (MSE) on the training set.
        - 'mse_test': The mean squared error (MSE) on the test set.
    """

    X = model_config.X
    y = model_config.y
    X_train = model_config.X_train
    y_train = model_config.y_train
    X_test = model_config.X_test
    y_test = model_config.y_test

    # Use optimizer for hyperparameter tuning
    optimizer.fit(X, y)

    # Get the best hyperparameters
    best_params = optimizer.best_params_

    # Train the model with the best hyperparameters
    final_model = model_class(**best_params)

    # Evaluate the model using cross-validation and calculates the mean
    cv_scores = cross_val_score(final_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    mse_mean_cv = np.mean(cv_scores)

    # Train the final model on the entire training set
    final_model.fit(X_train, y_train)

    # Evaluate the final model on the test set
    y_pred_test = final_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)

    output = {
        'params': best_params,
        'model': final_model,
        'mse_mean_cv': mse_mean_cv,
        'mse_test': mse_test
    }

    return output