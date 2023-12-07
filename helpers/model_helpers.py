from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

def two_step_hyperparameter_tuning(model_class, model_config, param_grid):
    """
    Perform two-step hyperparameter tuning using Random Search followed by Grid Search.

    Parameters:
    - model_class (class): The machine learning model class (e.g., SVR).
    - model_config (object): An object containing training and test data (X_train, y_train, X_test, y_test).
    - param_grid (dict): The hyperparameter grid to search.

    Returns:
    dict: A dictionary containing the results of the hyperparameter tuning:
        - 'params': Best hyperparameters from Grid Search.
        - 'model': Final trained model with the best hyperparameters.
        - 'mse_mean_cv': Mean squared error obtained from cross-validation on the training set.
        - 'mse_test': Mean squared error obtained on the test set.
    """

    X_train = model_config.X_train
    y_train = model_config.y_train
    X_test = model_config.X_test
    y_test = model_config.y_test

    random_search = RandomizedSearchCV(model_class, param_distributions=param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=5, random_state=42)

    # Use Random Search as first step of the hyperparameter tuning
    random_search.fit(X_train, y_train)

    # Get the best hyperparameters from Random Search
    best_params_random: list = random_search.best_params_

    # Use the best hyperparameters from Random Search as initial values for Grid Search
    grid_search_params: dict = {
        key: [value] for key, value in best_params_random.items()
    }

    grid_search = GridSearchCV(model_class, grid_search_params, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters from Grid Search
    best_params_grid: list = grid_search.best_params_

    # Train the final model with the best hyperparameters from Grid Search
    final_model = grid_search.best_estimator_

    # Evaluate the model using cross-validation and calculates the mean
    cv_scores: list = cross_val_score(final_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    mse_mean_cv: float = np.mean(cv_scores)

    # Train the final model on the entire training set
    final_model.fit(X_train, y_train)

    # Evaluate the final model on the test set
    y_pred_test = final_model.predict(X_test)
    mse_test: float = mean_squared_error(y_test, y_pred_test)

    output: dict = {
        'params': best_params_grid,
        'model': final_model,
        'mse_mean_cv': mse_mean_cv,
        'mse_test': mse_test
    }

    return output

